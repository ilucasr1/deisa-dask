# =============================================================================
# Copyright (C) 2026 Commissariat a l'energie atomique et aux energies alternatives (CEA)
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the names of CEA, nor the names of the contributors may be used to
#   endorse or promote products derived from this software without specific
#   prior written  permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# =============================================================================

import asyncio
import logging
import uuid
from collections import deque, defaultdict
from numbers import Number
from typing import Any, Iterator, List, Dict, Optional, Union, Deque

import numpy as np
from dask.tokenize import tokenize
from deisa.core import validate_arrays_metadata, IBridge, ICommunicator
from distributed import Queue, Client
from distributed.protocol import to_serialize
from distributed.utils_comm import scatter_to_workers
from tlz import valmap

from deisa.dask.deisa import FEEDBACK_QUEUE_PREFIX, CLIENT_KEY
from deisa.dask.handshake import Handshake
from deisa.dask.utils import get_client

logger = logging.getLogger(__name__)

from deisa.dask.bridge import Bridge

import pdi
import dask
import time
import json
import threading
from mpi4py import MPI
from typing import overload
from distributed import Worker

from distributed.compatibility import PeriodicCallback


class LBBridge(Bridge):
    def __init__(self, comm: ICommunicator, arrays_metadata: Dict[str, Dict], *args, **kwargs):

        super().__init__(comm, arrays_metadata, *args, **kwargs)
        self.round_robin_incr = 0
        self.injected_worker = None

        self.rank = self.comm.Get_rank()

        self.scheduler_address = None

        if self.rank == 0:
            self.client.run_on_scheduler(LBBridge._init_monitor)
            self.scheduler_address = self.client.scheduler.address

        self.scheduler_address = self.comm.bcast(self.scheduler_address, root=0)
        self.worker_thread = None
        self.loop = None

    def safe_send(self,
            array_name: str | List[str],
            chunk: np.ndarray | List[np.ndarray],
            timestep: int,
            arrays: dict[str,int] = None,
            *args,
            **kwargs):

        if arrays is not None:
            array_name = list(arrays.keys())
            chunk = list(arrays.values())

            self.safe_send(array_name, chunk, timestep, *args, **kwargs)
            return

        elif isinstance(array_name, List):
            assert isinstance(chunk, List) and len(array_name) == len(chunk)

            for i in range(len(array_name)):
                self.safe_send(array_name[i], chunk[i], timestep, multiple_sends=True, *args, **kwargs)

            self._await_worker()
            return

        to_send = chunk.nbytes
        safety_net = 6

        size = self.comm.Get_size()
        rank = self.comm.Get_rank()

        def filter_workers(workers):
            nonlocal safety_net

            avail_workers = None

            avail_workers_dict = dict()
            extra_workers = []
            # Requirement : 4 times the number of bytes to send
            required_nbytes = safety_net * to_send

            #print(f"[WorkerFilter-{rank}] Required nbytes for array {array_name} {required_nbytes}.",   flush=True)
            #print(f"[WorkerFilter-{rank}] Size {size}.", flush=True)

            for addr, info in workers.items():
                #print(f"[WorkerFilter-{rank}] Worker {addr} has {int(info['memory_limit'] - info['metrics']['memory'])} available bytes", flush=True)

                avail_chunks = int(info['memory_limit'] - info['metrics']['memory']) // required_nbytes
                #print(f"[WorkerFilter-{rank}] Worker {addr} has {avail_chunks} available chunks", flush=True)

                avail_workers_dict[addr] = size if (avail_chunks > size) else avail_chunks 
                if info["resources"].get("manual", 0):
                    extra_workers.append(addr)

            #print(f"[WorkerFilter-{rank}] Available workers {avail_workers_dict}.",   flush=True)

            addr_queue = deque((addr, count) for addr, count in avail_workers_dict.items())
            avail_worker_chunks = []

            while addr_queue:
                addr, count = addr_queue.popleft()
                if count > 0:
                    avail_worker_chunks.append(addr)    
                if count > 1:
                    addr_queue.append((addr, count - 1))

            if len(avail_worker_chunks) < size and extra_workers == []:
                print(f"[WorkerFilter-{rank}] Not enough space to send the data. Start new workers injection", flush=True)
                self._worker_inject()
                print(f"[WorkerFilter-{rank}] Injection finished", flush=True)

                updated_workers = None
                if rank == 0:
                    updated_workers = self.client.scheduler_info(n_workers=-1)["workers"]
                    print(f"[Worker_filter-{rank}] Updated workers {updated_workers.keys()}.", flush=True)
                updated_workers = self.comm.bcast(updated_workers, root=0)  
                return filter_workers(updated_workers)

            print(f"[WorkerFilter-{rank}] Available worker chunks: {len(avail_worker_chunks)} with {len(extra_workers)} extra workers", flush=True)

            if len(avail_worker_chunks) < size:
                print(f"[WorkerFilter-{rank}] Error, retry with smaller treshold", flush=True)
                safety_net -= 1
                if safety_net == 0:
                    print(f"[WorkerFilter-{rank}] No space, abort", flush=True)
                    return []
                return filter_workers(workers)

            res = avail_worker_chunks[(self.round_robin_incr + rank) % len(avail_worker_chunks)]
            self.round_robin_incr += 1 

            print(f"[bridge_send-{rank}] Sending {array_name} to worker: {res}.", flush=True)
            return [res]

        self.send(array_name, chunk, timestep, update_workers=True, filter_workers=filter_workers, *args, **kwargs)

        if 'multiple_sends' in kwargs and kwargs['multiple_sends']:
            return

        self._await_worker()


    def _worker_inject(self):
        size = self.comm.Get_size()
        rank = self.comm.Get_rank()

        async def start_worker():
            self.injected_worker = await Worker(
                                self.scheduler_address,
                                name=f"bridge_worker_{self.id}",
                                local_directory="/tmp",
                                memory_limit="10GiB",
                                memory_target_fraction=False,
                                memory_spill_fraction=False,
                                resources={"manual":1}
                                )


        print(f"[LBBridge-{rank}] Injection starting\n")
        
        self.loop = asyncio.new_event_loop()

        #def thread_main():
        #    asyncio.set_event_loop(self.loop)
        #    self.loop.run_forever()

        self.worker_thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.worker_thread.start()

        asyncio.run_coroutine_threadsafe(start_worker(), self.loop)

        if rank == 0:

            prior_nworkers = 0
            workers = self.client.scheduler_info(n_workers=-1)["workers"]
            for w in workers.values():
                if "manual" not in w.get("resources", {}):
                    prior_nworkers +=1
            
            print(f"[LBBridge-{rank}] Waiting for {prior_nworkers+size} workers", flush=True)
            self.client.wait_for_workers(n_workers=prior_nworkers+size)

            time.sleep(0.5)
            #self.client.register_plugin(WorkerRestartPlugin(), "WorkerRestartPlugin")
            self.client.run_on_scheduler(LBBridge._start_monitor)

        self.comm.Barrier()
        print(f"[LBBridge-{rank}] Injection complete.", flush=True)


    def _await_worker(self):
        if self.injected_worker is None:
            return
        rank = self.comm.Get_rank()

        #import concurrent.futures

        async def await_worker_retirement():
            print(f"[LBBridge-{rank}] Injected worker waiting for retirement.", flush=True)
            #future = asyncio.run_coroutine_threadsafe(self.injected_worker.finished(), self.loop)
            #future.result()
            await self.injected_worker.finished()
            print(f"[LBBridge-{rank}] Injected worker retirement complete.", flush=True)

            self.injected_worker = None
        
        future = asyncio.run_coroutine_threadsafe(await_worker_retirement(), self.loop)
        future.result()

        self.loop.call_soon_threadsafe(self.loop.stop)
        #while self.loop.is_running():
        #    print("Loop is runnning")
        #    time.sleep(0.1)
        self.worker_thread.join()
        self.loop.call_soon_threadsafe(self.loop.close)
        self.loop = None
        self.worker_thread = None

        self.comm.Barrier()

        if rank == 0:
            self.client.run_on_scheduler(LBBridge._stop_monitor)


    @staticmethod
    def _init_monitor(dask_scheduler):

        print(f"[LBCallback] Initialising load balance callback", flush=True)

        lock = asyncio.Lock()

        async def monitor():
            if lock.locked():
                return 

            async with lock:
                injected_workers = [addr for addr, info in dask_scheduler.workers.items() if info.resources.get("manual")]
#                await dask_scheduler.retire_workers(injected_workers,
#                                                    close_workers=False,
#                                                    remove=False)

                staying_workers = list(dask_scheduler.workers.keys() - set(injected_workers))

                injected_mem_managed = 0
                base_mem_available = 0
                
#                already_removed = []


                for addr, info in dask_scheduler.workers.items():
#                    print(f"ADDR : {addr}\nINFO : {info}", flush=True)
                    mem_used = info.metrics["memory"]
                    mem_managed = info.nbytes
                    mem_limit = info.memory_limit

                    if addr in injected_workers:
#                        if info.status == "closing_gracefully":
#                            await dask_scheduler.remove_worker(
#                                address=addr,
#                                stimulus_id="manual-cleanup",
#                                close=True
#                            )
#                            already_removed.append(addr)
#                            continue
                            
#                        print(f"[LBPCallback] Worker {addr}: used {mem_managed} out of {mem_limit}.", flush=True)
                        injected_mem_managed += mem_managed
#                        print(
#                            f"{addr} managed={info.nbytes/2**20:.1f} MiB "
#                            f"keys={len(info.has_what)}",
#                            flush=True,
#                        )
#
#                        for ts in list(info.has_what)[:10]:
#                            print(
#                                f"  {ts.key} -> {ts.nbytes/2**20:.1f} MiB",
#                                flush=True,
#                            )
#                            print("who_wants:", [ww for ww in ts.who_wants] if ts.who_wants is not None else [])
#                            print("key:", ts.key)
#                            print("dependencies:", [d.key for d in ts.dependencies])
                    else:
                        base_mem_available += (mem_limit - mem_used)
                
#                injected_workers = [w for w in injected_workers if w not in already_removed]
                
                if 3*injected_mem_managed < base_mem_available:
                    print(f"[LBCallback] Stopping load balance callback", flush=True)
                    dask_scheduler.periodic_callbacks["monitor"].stop()

                    print(f"[LBPCallback] Trigger rejection", flush=True)
                    await dask_scheduler.retire_workers(injected_workers,
                                                        close_workers=True,
                                                        remove=True)

        pc = PeriodicCallback(monitor, 5000) # every 5 seconds
        dask_scheduler.periodic_callbacks["monitor"] = pc


    @staticmethod
    def _start_monitor(dask_scheduler):
        print(f"[LBCallback] Start monitor callback", flush=True)
        pc = dask_scheduler.periodic_callbacks['monitor']
        if not pc._running: 
            pc.start()


    @staticmethod
    async def _stop_monitor(dask_scheduler):
        print(f"[LBCallback] Stop monitor", flush=True)
        pc = dask_scheduler.periodic_callbacks['monitor']
        if pc is not None and pc._running: 
            pc.stop()

        #print(f"[LBPCallback] Trigger rejection", flush=True)
        injected_workers = [addr for addr, info in dask_scheduler.workers.items() if info.resources.get("manual")]
        #print(f"[LBPCallback] Workers to reject: {injected_workers}", flush=True)
        #await dask_scheduler.remove_workers(injected_workers, close=True) 
        #await dask_scheduler.retire_workers(injected_workers, close_workers=True) 
        for addr in injected_workers:
            if addr in dask_scheduler.workers:
                print(f"[LBCallback] Attempting to remove worker {addr}.", flush=True)
                await dask_scheduler.remove_worker(addr, stimulus_id="end_program", close=True)

        print(f"[LBCallback] Rejection complete.", flush=True)
