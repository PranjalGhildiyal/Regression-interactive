def get_var_name(var_value):
    for name, value in globals().items():
        if value is var_value:
            return name
    return None