# This is an example of printing a message

import GN4IP
import time

# Print lines
GN4IP.utils.printLine(10)
GN4IP.utils.printLine()

# Print a timed message
tstart = time.time()
message1 = "The timer just started!"
GN4IP.utils.timeMessage(tstart, message1)
time.sleep(3)
message2 = "This is a message after sleeping for 3 seconds."
GN4IP.utils.timeMessage(tstart, message2)