## Index

- [Modules](#modules)
- [Functions](#functions)
- [Remember](#remember)

# MODULES

Import a module: `import module_name` _(PREFERRED for readability)_  
Import everything from module and avoid the dot syntax: `from module_name import *`

```py
# PREFERRED VERSION
import random
print(random.randint(0, 10)) #4

#LESS READABLE
from random import *
print(randint(0, 10)) # 4
```

## random

### randint() function

```py
import random

print(random.randint(0, 10)) # 4
```

🔝 **[INDEX](#index)** 🔝

## sys

Aids in interacting with the Python interpreter, and the system.

### exit() function

```py
import sys

print('Hello') # Hello
sys.exit()
print('Goodbye') # (Will not print)
```

## pyperclip (3rd party)

Reads from and writes text to the clipboard.

```terminal
pip install pyperclip
```

### copy() and paste() functions

```py
import pyperclip

pyperclip.copy('Hello world!')
pyperclip.paste()

# 'Hello world!'
```

🔝 **[INDEX](#index)** 🔝

# FUNCTIONS

## def

```py
def hello:
    print('Hello')
    print('Hey')
    print('Hi')

hello()
hello()

# Hello
# Hey
# Hi
# Hello
# Hey
# Hi
```

🔝 **[INDEX](#index)** 🔝

# REMEMBER

## F-Strings (... are template literals)

```py
a = 10
b = 5
result = a + b
print(f"The sum of {a} and {b} is {result}.")

# The sum of 10 and 5 is 15.
```

🔝 **[INDEX](#index)** 🔝
