from concurrent.futures import ThreadPoolExecutor

from brrr import cuext


def main() -> None:
    with ThreadPoolExecutor(len(cuext.SOURCES)) as pool:
        list(pool.map(cuext.extension, cuext.SOURCES))


if __name__ == "__main__":
    main()
