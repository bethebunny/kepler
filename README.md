# Kepler

Wish you could see stuff like this about your program?

<img width="1227" alt="image" src="https://github.com/user-attachments/assets/aa60de68-2648-4794-a29c-873365bc077b">


Kepler meticulously tracks your program, and creates simply and easily readable reports to help you understand what they're doing.

Kepler _is not_ a replacement for a good profiling tool, nor is it necessarily a great production implementation tool. Kepler is designed to be that go-to tool in your toolbelt for quick and dirty measurements of your programs.

## Installing Kepler

```bash
pip install kepler
```

## Kepler in action

The first thing you should do with Kepler is annotate a function or two you want to time with `@kepler.time`, and then add a `kepler.report()` call to your amin function.

Here's the script that produced the screenshot above:

```python
import kepler, random, time

@kepler.time("do some stuff")
def do_some_stuff():
    split = kepler.stopwatch("watch")
    for i in kepler.time("loop", range(20)):
        with kepler.time("sleep"):
            time.sleep(random.random() / 100)
        if i % 2 == 1:
            with kepler.time("overhead"):
                split("odd")
        else:
            with kepler.time("overhead"):
                split("even")

@kepler.time
def main():
    with kepler.time("sloooow"):
        time.sleep(0.3)
    do_some_stuff()

main
kepler.report()
```
