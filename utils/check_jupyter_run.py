def check_jupyter_run():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False