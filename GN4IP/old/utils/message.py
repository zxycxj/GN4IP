# Functions for displaying print output

import time
 
# Display horizontal lines
def printLine(N=50):
    
    # Make the string of equal signs and print it with the # signs
    line = "=" * (N-2)
    print("#"+line+"#")
    

# Display the time with a message
def timeMessage(tstart, message):
    
    # Get the time
    if tstart == 0:
        hrs  = 0
        mins = 0
        sec  = 0
    else:
        tend = time.time() - tstart
        hrs  = int( tend//3600)
        mins = int((tend -3600*hrs)//60)
        sec  = int((tend -3600*hrs - 60*mins) // 1)

    # Print the message
    print('# ({:02d}:{:02d}:{:02d}) {}'.format(hrs, mins, sec, message))
    
