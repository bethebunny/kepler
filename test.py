import random
import time

import kepler


@kepler.time("do some stuff")
def do_some_stuff():
    split = kepler.stopwatch()
    for i in kepler.time("loop", range(20)):
        with kepler.time("sleep"):
            time.sleep(random.random() / 100)
        if i % 2 == 1:
            with kepler.time("overhead"):
                split("odd")


@kepler.time
def main():
    with kepler.time("sloooow"):
        time.sleep(0.3)
    do_some_stuff()


if __name__ == "__main__":
    main()
    kepler.report()
