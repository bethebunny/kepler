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

main()
kepler.report()
```

### Adding custom timers

Custom timers may be implemented with the `measurement` decorator.
This works similarly to Python's `contextlib.contextmanager` decorator:

- The decorated function should be a generator
- It should do any setup it needs to do, and then `yield` _exactly once_
- `yield` call corresponds exactly to the code to be measured
- It should _return an `Event`_ to be added to kepler's log

```python
import time
import kepler
import torch

@kepler.measurement
def time_gpu():
    start_time = time.time_ns()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    yield
    end.record()
    torch.cuda.synchronize()
    return kepler.TimingEvent(start_time, start.elapsed_time(end) * 1e6)


with time_gpu("matmul"):
    _ = torch.rand([2, 2]) @ torch.rand([2, 2])

kepler.report()
```

## Roadmap

### ✅ Changelog

- [Custom timing measurements](https://github.com/bethebunny/kepler?tab=readme-ov-file#adding-custom-timers)
- Import and export reports as json
- Report directly from json -- try `python -m kepler.report < tests/data/simple_log.json`
- Testing for units, json import and export

### 🔜 Up next

- Track system metrics

### 🌈 Before 1.0

- Export traces to pandas
- Flamegraphs
- Integrate with open-telemetry
- Track and report other metrics besidings timings
- Thorough unit testing
- Docs
- Examples
- Logo
