def show_params(args):
    print("< Config Params >")
    for key, value in args.items():
        print(" - {}: {}".format(key, value))

    print("---")
