# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 09:28:51 2021

@author: imrul
"""

import asyncio

async def main():
    print('Hello ...')
    await asyncio.sleep(1)
    print('... World!')

# Python 3.7+
# asyncio.run(main())

try:
    loop = asyncio.get_running_loop()
except RuntimeError:  # if cleanup: 'RuntimeError: There is no current event loop..'
    loop = None

if loop and loop.is_running():
    print('Async event loop already running')
    tsk = loop.create_task(main())
    # # ^-- https://docs.python.org/3/library/asyncio-task.html#task-object
    # tsk.add_done_callback(                                          # optional
    #     lambda t: print(f'Task done: '                              # optional
    #                     f'{t.result()=} << return val of main()'))  # optional (using py38)
else:
    print('Starting new event loop')
    asyncio.run(main())