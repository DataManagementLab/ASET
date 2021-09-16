"""
The main entry point into Ad-hoc Structured Exploration of Text Collections User Interface (ASET UI).

Run this script to execute ASET UI, which is a graphical user interface for ASET.

ASET extracts information nuggets (extractions) from a collection of documents and matches them to a list of
user-specified attributes. Each document corresponds with a single row in the resulting table.
"""

if __name__ == "__main__":
    # set up logging
    import logging.config

    logging.config.fileConfig("logging.conf")
    logger = logging.getLogger()
    logger.info("Starting application.")

    # start the application
    import sys
    from aset_ui import app

    sys.exit(app.run(sys.argv))
