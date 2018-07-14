# Word Neural Associative Memory

WIP

# Setup

## Download pre-trained word vector

- http://www.cl.ecei.tohoku.ac.jp/~m-suzuki/jawiki_vector/

```
$ cp entity_vector.model.bin ./model.bin
```

## Install dependencies

```
$ pip install -r requirements.txt
```

# Run word vector server

```
$ python word_vector_server.py
```

# Train

```
$ python main.py --train
```

# Test

```
$ python main.py
```

