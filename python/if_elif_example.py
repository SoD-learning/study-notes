# print('What is your name?')
# name = input()
# print('How old are you?')
# age = int(input())

# if name == 'Alice':
#     print('You are Alice.')
# elif age < 12:
#     print('You are not Alice because you are too young.')
# elif age > 2000:
#     print('You are not Alice because she is not an immortal vampire.')
# elif age > 100:
#     print('You are not Alice because she is not a granny.')

def whoTheFuckIsAlice():
    print('What is your name?')
    name = input()
    print('How old are you?')
    age = int(input())

    if name == 'Alice':
        print('You are Alice.')
    elif age < 12:
        print('You are not Alice because you are too young.')
    elif age > 2000:
        print('You are not Alice because she is not an immortal vampire.')
    elif age > 100:
        print('You are not Alice because she is not a granny.')
    print()

playAgain = 'yes'
while playAgain == 'yes' or playAgain == 'y':

    whoTheFuckIsAlice()
    
    print('Do you want to play again? (yes or no)')
    playAgain = input()