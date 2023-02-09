# -*- coding: utf-8 -*-
"""
Author -- Michael Widrich, Andreas Schörgenhumer
Contact -- schoergenhumer@ml.jku.at
Date -- 02.08.2022

################################################################################

The following copyright statement applies to all code within this file.

Copyright statement:
This material, no matter whether in printed or electronic form, may be used for
personal and non-commercial educational use only. Any reproduction of this
manuscript, no matter whether as a whole or in parts, no matter whether in
printed or in electronic form, requires explicit prior acceptance of the
authors.

################################################################################

In this file, we will see how we can start external programs in the background
using the "subprocess" module and how to call functions or external programs in
a parallel fashion using the "multiprocessing" module. Using the modules shown
in this unit, Python can be used as powerful alternative to shell-/bash-scripts
to call and communicate with other programs and to use multiple processes, while
being largely independent of the operating system.
"""

################################################################################
# subprocess - spawning and managing subprocesses
################################################################################

# The "subprocess" module allows you to execute other programs in the
# background/in parallel, catch their output and error messages, communicate
# with them, and manage them. There are many (partly redundant) functions
# available. Only a small part is shown here, see the documentation for a full
# description of the entire functionality.
# https://docs.Python.org/3/library/subprocess.html

import subprocess

#
# Running programs in the background
#

# The class "Popen" allows you to run programs in background, i.e., the program
# call is initiated, but you return immediately and execute the rest of your
# Python program code. The minimum calling signature is providing the executable
# together with string arguments as a sequence, i.e, the basic usage looks like:
# subprocess.Popen(["some_program", "argument1", "argument2"])
# However, the interface is much more rich in options. For more details, see
# https://docs.python.org/3/library/subprocess.html#subprocess.Popen

# Example of calling the "ping" program. Note that this program is slightly
# different in Linux (e.g., Ubuntu), so you might have to execute something like
# subprocess.Popen(["ping", "www.jku.at", "-c", "4"])
# to avoid pinging indefinitely (on Windows, the ping count is 4 by default)
p = subprocess.Popen(["ping", "www.jku.at"])  # Will not wait for the program to finish

# The following code will wait 2sec for the program to terminate or raise a
# "TimeoutExpired" exception otherwise:
try:
    p.wait(timeout=2)
except subprocess.TimeoutExpired:
    # What should we do if the process does not finish in time? We could kill
    # it, give it more time, etc., but for now, we simply kill it
    print("Process could not finish in time - killing it!")
    p.kill()

# If we want to access the output of the program, we have to use pipes. Here, we
# will get the standard output (stdout) and the error messages (stderr) during
# the execution of the program in the background (the output is redirected, so
# there will not be any of it visible in the standard console). The output will
# be returned as a raw bytes, which will have to be decoded according to the
# character encoding the program used. For instance here, the "ping" program on
# Windows uses the OEM character encoding, so you would have to decode the
# returned bytes with ".decode('oem')". Alternatively, you can directly specify
# the "encoding" parameter when creating the "Popen" object.
p = subprocess.Popen(["ping", "www.jku.at"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# This will wait 15sec for the program to terminate or raise a "TimeoutExpired"
# exception. Output and error messages will be sent to variables "outs", "errs".
try:
    outs, errs = p.communicate(timeout=15)
    print(f"outs: {outs}")
    print(f"errs: {errs}")
except subprocess.TimeoutExpired:
    print("Process could not finish in time - killing it!")
    p.kill()

# You can use the with statement, which automatically (properly) closes standard
# file descriptors and waits for the process, i.e., code below the statement is
# not executed until the process terminates (either normally or, e.g, by us).
with subprocess.Popen(["ping", "www.jku.at"], stdout=subprocess.PIPE, stderr=subprocess.PIPE) as p:
    try:
        outs, errs = p.communicate(timeout=1)
        print(f"outs: {outs}")
        print(f"errs: {errs}")
    except subprocess.TimeoutExpired:
        # Terminate the process and get all output that was produced until then
        p.kill()
        outs, errs = p.communicate()
        print(f"TIMEOUT outs: {outs}")
        print(f"TIMEOUT errs: {errs}")

#
# Simplified program execution
#

# "Popen" enables full control. However, it might often be the case that a
# simple program call would be enough. For such cases, the convenience function
# "run" exists. The interface is similar, but the program is waited for, and a
# "CompletedProcess" result object is returned instead. Most basic usage:
# result = subprocess.run(["some_program", "argument1", "argument2"])
# The "run" function is also much more powerful and supports timeouts, output
# and error piping and encoding settings. For more details, see
# https://docs.python.org/3/library/subprocess.html#subprocess.run
# https://docs.python.org/3/library/subprocess.html#subprocess.CompletedProcess

# Example of calling the Python executable itself. If this call does not work
# (e.g., "python" cannot be found due to missing environment variable entries),
# you can always specify the fully qualified path to the executable/program. For
# Python, there is also a shortcut if you simply want to call the same Python
# executable this script here was executed with. This can be done via the "sys"
# module: Import "sys" and the use "sys.executable", which gives you the path
# to the executable.
result = subprocess.run(["python", "--help"])
print(result.returncode)

# Simplified example of running the "ping" program and capturing all output:
result = subprocess.run(["ping", "www.jku.at"], capture_output=True)
print(result.returncode)
print(f"outs: {result.stdout}")
print(f"errs: {result.stderr}")


################################################################################
# multiprocessing - easy and powerful parallelization
################################################################################

# The "multiprocessing" module provides utilities for easy and safe distribution
# of a task to a pool of worker processes. Only a small part is shown here, see
# the documentation for a full description of the entire functionality.
# https://docs.python.org/3/library/multiprocessing.html

# We will only use the "Pool" class in this code example. Note on the code
# execution: To properly see how all the following examples work, copy each one
# into a separate file, and then run this file.
from multiprocessing import Pool
import time  # For artificially making our function calls take longer


def f(x):
    # This is the function we want to run in parallel. You could also call
    # something like "subprocess.run" here to run non-Python programs. In
    # addition, we call "time.sleep" to wait for some time, so we can actually
    # see some differences when we change the number of used worker processes.
    time.sleep(0.1)
    return x * x


# Now, we will create a pool of 4 worker processes in the background. Then, the
# function "f" will be applied to each element of some list that we specified,
# using all 4 worker processes in parallel.

# This if-condition is important to only execute the code in the main process,
# it will not work otherwise (in the background, the file is imported again, so
# the import must be done safely without executing the same multiprocessing code
# all over again).
if __name__ == "__main__":
    # This will be the list we want to process:
    arguments = list(range(100))
    # Here, we create the pool of 4 worker processes. Using the with statement,
    # we ensure that the pool is properly shut down afterwards.
    with Pool(4) as p:
        # You can use "Pool.map" to apply a function to an iterable of arguments
        # in a parallel fashion. "Pool.map" will automatically manage the
        # processes and work through the list in parallel.
        # https://docs.python.org/3/library/multiprocessing.html#multiprocessing.pool.Pool.map
        pool_returns = p.map(f, arguments)
    print(pool_returns)

#
# Multiprocessing using a dynamic iterable
#

# "Pool.map" will process the list of arguments at once and store the result in
# a list. This can consume large amounts of memory, and we do not see the status
# of the processes. The "Pool.imap" method addresses these issues. It returns a
# dynamic iterable over which we can iterate in a loop. As with "map", the pool
# of workers is used to process the arguments, but we can access the results as
# soon as they are ready.
if __name__ == "__main__":
    arguments = list(range(100))
    with Pool(4) as p:
        # By using the "Pool.imap" method, we dynamically get results as soon as
        # they are ready, instead of having to wait for a list of all results.
        # https://docs.python.org/3/library/multiprocessing.html#multiprocessing.pool.Pool.imap
        # Note that "imap" returns the individual results in the same order as the
        # arguments. A faster but unordered version is the "imap_unordered" method.
        for return_value in p.imap(f, arguments):
            print(return_value)


# If you have a function with more than one parameter, use "starmap" instead of
# "map". With "starmap", the arguments are expected to be iterables themselves,
# which are then unpacked and passed to the function.
def g(x, y):
    time.sleep(0.1)
    return x * y


if __name__ == "__main__":
    # If we were to use "map" below, each element (here: tuple) would be passed
    # as a single argument, e.g., g((1, 2)). If we use "starmap" instead, each
    # element (here: tuple) is expected to be an iterable and is unpacked when
    # passing it to the function, e.g., g(*(1, 2)) -> g(1, 2) -> g(x=1, y=2)
    # https://docs.python.org/3/library/multiprocessing.html#multiprocessing.pool.Pool.starmap
    arguments = [(1, 2), (3, 4), (5, 6)]
    with Pool(2) as p:
        pool_returns = p.starmap(g, arguments)
    print(pool_returns)

# Note: Out of the box, keyword unpacking is not supported, but you can write a
# small wrapper function that does the unpacking for you, e.g.:
#
# def g_wrapped(my_dict):
#     g(**my_dict)
#
# Afterwards, simply use "map" again with a list of dictionaries, e.g.:
#
# arguments = [dict(x=1, y=2), dict(x=3, y=4), dict(x=5, y=6)]
# pool.map(g_wrapped, arguments)

#
# Asynchronous execution
#

# If you do not need or want to handle the results of the parallelized function
# calls immediately, then there are also asynchronous versions of some methods
# available. For example, "Pool.map_async" will not return the results of the
# function as a list but rather an "AsyncResult" object instead. This object
# allows "access" to the (potentially still running) function calls. Since this
# is an advanced topic, we will not cover it here. For more details, see
# https://docs.python.org/3/library/multiprocessing.html#multiprocessing.pool.Pool
# https://docs.python.org/3/library/multiprocessing.html#multiprocessing.pool.AsyncResult
    
#
# Practical example
#

# Let's combine some of the tools we have seen so far with the "tqdm" module,
# which creates progress bars (you probably have to install this module since it
# is not part of plain Python, e.g., via "pip install tqdm"). Assume we have
# some files and want to apply an external program to them. Furthermore, we want
# to do some post-processing using our function "some_postprocessing". We can
# easily apply our pipeline of calling the external program and using our
# post-processing in parallel using the "multiprocessing" module:


def some_postprocessing(preprocessed_input):
    # This function pretends to do some post-processing. Actually, we just do
    # some calculations that require CPU power for some time
    a = 0
    for i in range(int(1e7)):
        a += i / (1 + len(preprocessed_input))
    return a


def pipeline(filename):
    # This function pretends to call some program that processes the file with
    # name "filename". Actually, we just call the program "ping" again (here,
    # with Windows option "-t" to indicate pinging indefinitely; in Linux (e.g.,
    # Ubuntu), this is the default when calling "ping" with the address alone).
    # After 1 second, we terminate this program call and proceed normally.
    with subprocess.Popen(["ping", "www.jku.at", "-t"], stdout=subprocess.PIPE, stderr=subprocess.PIPE) as p:
        try:
            outs, errs = p.communicate(timeout=1)
        except subprocess.TimeoutExpired:
            p.kill()
            outs, errs = p.communicate()
    
    # Use the output of the external program and apply some post-processing
    final_output = some_postprocessing(preprocessed_input=outs)
    
    # Lastly, return the filename and the final output
    return filename, final_output


# This will provide us with a nice progressbar. For more details, see
# https://tqdm.github.io/ (not necessary, only if you are interested)
from tqdm import tqdm

if __name__ == "__main__":
    arguments = list(range(200))
    with Pool(4) as p:
        results = []
        for result_tuple in tqdm(p.imap(pipeline, arguments), total=len(arguments), desc="processing files"):
            results.append(result_tuple)
    print(f"Collected {len(results)} results")

# The above code is a good example of differences in CPU utilization. Without
# changing the code, we see that our 4 worker processes do not lead to a 100%
# CPU utilization (per core, assuming that you have at least 4 cores/hardware
# threads available). This is because we call the "ping" program, which is not
# CPU-heavy but network-IO-heavy. In such a case, we could increase the number
# of processes even beyond the actually available physical CPU cores and still
# (potentially) get a speed-up (of course, this depends on the actual task and
# the used hardware). To shift our example towards more CPU-bound processing,
# you can remove the "ping" call and just invoke "some_postprocessing" (you
# can choose some arbitrary argument instead of "outs"). Run the program again
# and see that the CPU utilization will be much higher now. Also try to increase
# the number of pool processes (according to your hardware) to see the effects
# on the CPU utilization as well as on the total run time of the program.
