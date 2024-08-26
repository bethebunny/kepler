# Kepler
Kepler meticulously tracks your program, and creates simply and easily readable reports to help you understand what they're doing.

Kepler _is not_ a replacement for a good profiling tool, nor is it necessarily a great production implementation tool. Kepler is designed to be that go-to tool in your toolbelt for quick and dirty measurements of your programs.

## Installing Kepler

```bash
pip install kepler
```

## Kepler in action

The first thing you should do with Kepler is annotate a function or two you want to time with `@kepler.time`, and then add a `kepler.report()` call to your amin function.

Here's a short example showing off some of the various features, along with the output of `kepler.report()`.

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

<img width="916" alt="image" src="https://github.com/user-attachments/assets/84cf4300-32cb-476a-8a85-ea1f30ce6272">
