import random
import time

import kepler


@kepler.time("do some stuff")
def do_some_stuff():
    for i in range(10):
        if i % 2 == 1:
            kepler.split("odd")
        with kepler.time("in loop"):
            time.sleep(random.random() / 100)


@kepler.time
def main():
    with kepler.time("sloooow"):
        time.sleep(0.0001)
    kepler.split("startup")
    do_some_stuff()


if __name__ == "__main__":
    main()
    kepler.report()
