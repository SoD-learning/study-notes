# name = ''
# while name != 'your name': # heh...
#     print('Please type your name.')
#     name = input()
# print('Thank you!')

name = ''
while True: # This will ALWAYS be true because True evaluates to True
    print('Please type your name.')
    name = input()
    if name == 'your name': # This is what breaks the 'forever True' loop
        break
print('Thank you!')