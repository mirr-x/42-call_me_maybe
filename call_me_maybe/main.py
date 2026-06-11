""" Call Me Maybe Entry Point """

import logging

def main() -> None:
    """ Main function for Call Me Maybe """
    logging.info("Call Me Maybe started\n")
    logging.info("Program has Ended")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
        )

    try:
        main()
    except KeyboardInterrupt as e:
        logging.error("Program has been interrupted by the user: %s", e)
