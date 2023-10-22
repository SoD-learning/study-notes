spam = 0
while spam < 5:
    spam = spam + 1
    if spam == 3: # Perform action when spam is 3
        print('pizza')
        continue # Go back to the start of the while loop
    print('spam is ' + str(spam)) # str() converts int to str for concatenation