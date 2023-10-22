## Index

- [How To](#how-to)
- [When To](#when-to)
- [Formatting](#formatting)
- [Tricks](#tricks)
- [Remember](#remember)
- [Reference](#reference)

# HOW TO ...

## Convert string to int

```py
int('26') # => 26
```

🔝 **[INDEX](#index)** 🔝

# WHEN TO ...

## Convert string to int

```py
myAge = input() # => '21'
print('In a year, you will be ' + str(int(myAge) + 1) + ' years old') # => In a year, you will be 22 years old
```

🔝 **[INDEX](#index)** 🔝

# FORMATTING

## Booleans

```py
True # => True
False # => False
true # => NameError: name 'true' is not defined. Did you mean: 'True'?
false # => NameError: name 'false' is not defined. Did you mean: 'False'?
```

```bash
bool(0) # => False
bool(42) # => True
bool('Hello') # => True
bool('') # => False
```

🔝 **[INDEX](#index)** 🔝

# TRICKS

`shift + enter` executes the chosen line (i.e. cursor's location) in the terminal.

🔝 **[INDEX](#index)** 🔝

# REMEMBER

## Equality

```py
42 == '42' # => False (int and str always evaluate to False)
42.0 == 42 # => True (int and float evaluate to True)
```

## Truth tables

```py
True and True # => True
False and True # => False
False and False # => False
True and False # => False

True or True # => True
True or False # => True
False or False # => False

not True # => False
not False # => True
```

```py
myAge = 21
myPet = 'cat'
myAge > 20 and myPet == 'cat' # => True
```

## Loops

```py
# FOR
print('My name is')
for i in range(5):
    print('Jimmy Five Times ' + str(i))

# WHILE
print('My name is')
i = 0
while i < 5:
    print('Jimmy Five Times ' + str(i))
    i = i + 1

# => My name is
# => Jimmy Five Times 0
# => Jimmy Five Times 1
# => Jimmy Five Times 2
# => Jimmy Five Times 3
# => Jimmy Five Times 4
```

## Range

```py
# range(start_at, stop_before, increment)
range(10) # => 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
range(1, 6) # => 1, 2, 3, 4, 5
range(0, 10, 2) # => 0, 2, 4, 6, 8
range(5, -1, -1) # => 5, 4, 3, 2, 1, 0
```

```py
print('My name is')
for i in range(12, 16):
    print('Jimmy Five Times ' + str(i))

# => My name is
# => Jimmy Five Times 12
# => Jimmy Five Times 13
# => Jimmy Five Times 14
# => Jimmy Five Times 15
```

## Code Blocks

Code blocks are identified by a colon.

```py
if name == 'Alice': # `:` indicates that a code block follows below
    print('Hi Alice')
```

🔝 **[INDEX](#index)** 🔝

# REFERENCE

## Comparison Operators

| Operator | Meaning                  |
| -------- | ------------------------ |
| `==`     | Equal to                 |
| `!=`     | Not equal to             |
| `<`      | Less than                |
| `>`      | Greater than             |
| `<=`     | Less than or equal to    |
| `>=`     | Greater than or equal to |

## Boolean Operators

`and`  
`or`  
`not`

🔝 **[INDEX](#index)** 🔝
