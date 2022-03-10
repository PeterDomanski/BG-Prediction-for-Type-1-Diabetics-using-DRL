import gin
from absl import app, logging


def main(args):
    # parse config file
    gin.parse_config_file("config.gin")
    run()


def run():
    print("Not implemented yet")


if __name__ == '__main__':
    app.run(main)
