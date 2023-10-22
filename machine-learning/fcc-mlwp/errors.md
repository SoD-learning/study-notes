# Errors and Their Fixes

🛑 `ValueError: Can't convert non-rectangular Python sequence to Tensor.`

**CONTEXT:**

```python
rank2_tensor = tf.Variable(["one", "two"], ["one", "two", "three"], tf.string)
```

**FIX:**

```python
rank2_tensor = tf.Variable(["one", "two", "three"], ["one", "two", "three"], tf.string)
```

**EXPLANATION:** Number of elements in grouped dimensions must be the same.
