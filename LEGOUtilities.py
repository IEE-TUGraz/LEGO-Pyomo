# Turns "k0001" into 1, "k0002" into 2, etc.
def k_to_int(k: str):
    return int(k[1:])


# Turns 1 into "k0001", 2 into "k0002", etc.
def int_to_k(i: int, digits: int = 4):
    return f"k{i:0{digits}d}"


# Turns "rp01" into 1, "rp02" into 2, etc.
def rp_to_int(rp: str):
    return int(rp[2:])


# Turns 1 into "rp01", 2 into "rp02", etc.
def int_to_rp(i: int, digits: int = 2):
    return f"rp{i:0{digits}d}"
