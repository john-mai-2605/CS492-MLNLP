def check_environment():

    try:
        import tensorflow
        print("tensorflow: {}".format(tensorflow.__version__))
    except ModuleNotFoundError:
        print("tensorflow: {}".format("FAIL"))

    try:
        import sklearn
        print("sklearn: {}".format(sklearn.__version__))
    except ModuleNotFoundError:
        print("sklearn: {}".format("FAIL"))

    try:
        import numpy
        print("numpy: {}".format(numpy.__version__))
    except ModuleNotFoundError:
        print("numpy: {}".format("FAIL"))

    try:
        import tqdm
        print("tqdm: {}".format(tqdm.__version__))
    except ModuleNotFoundError:
        print("tqdm: {}".format("FAIL"))

    import sys
    if sys.version_info.major == 3:
        print("python: {}".format("3"))
    else:
        print("python: {}".format("FAIL"))


if __name__ == '__main__':
    check_environment()
