# Test data generation
import random

MAX_INT = 4294967295  # 0 = 0.0.0.0 ; 4294967295 = 255.255.255.255
HALF_MAX_INT = MAX_INT // 2
MAX_MAC = 281474976710655  # 0 = 00:00:00:00:00:00 ; 281474976710655 = FF:FF:FF:FF:FF:FF

def gen_rand_ips(count: int):
    if count <= 0 or count > MAX_INT:
        print("bruh")
        return None
    
    print("Generating random IPs")
    res = set()
    # sample without replacement
    for ip in random.sample(range(MAX_INT+1), count):
        res.add(ip)

    return res

def gen_close_ips(count: int):
    if count <= 0 or count >= HALF_MAX_INT:
        print("bruh")
        return None
    print("Generating close IPs")
    
    # don't want too many 0s or 1s, so we'll randomly generate each bit
    # then we will grab values surrounding it
    start_ip = 0
    for _ in range(32):
        start_ip = (start_ip << 1) + random.randint(0, 1)
    
    res = set()
    res.add(start_ip)

    for i in range(count // 2):
        res.add((start_ip + i) % MAX_INT)
        res.add((start_ip - i) % MAX_INT)
    return res

def gen_edge_ips(count: int):
    if count <= 0 or count >= HALF_MAX_INT:
        print("bruh")
        return None
    print("Generating edge IPs")
    
    res = set()
    for i in range(count // 2):
        res.add(i)
        res.add(MAX_INT-i)
    return res

def gen_repeat_ips():
    # generates all the repeated IPs
    # e.g. 31.31.31.31, 250.250.250.250, etc.
    print("Generating repeat IPs")
    res = set()
    for i in range(255):
        ip = i
        ip = ((((ip << 8) + i) << 8) + i << 8) + i
        res.add(ip)
    return res

def gen_rand_macs(count: int):
    print("Generating random MACs")
    res = set()
    for mac in random.sample(range(MAX_MAC+1), count):
        res.add(mac)
    return res

def gen_close_macs(count: int):
    print("Generating close MACs")

    start_mac = 0
    for _ in range(48):
        start_mac = (start_mac << 1) + random.randint(0, 1)
    
    res = set()
    res.add(start_mac)

    for i in range(count // 2):
        res.add((start_mac + i) % MAX_MAC)
        res.add((start_mac - i) % MAX_MAC)
    return res

def gen_edge_macs(count: int):
    print("Generating edge MACs")
    
    res = set()
    for i in range(count // 2):
        res.add(i)
        res.add(MAX_MAC-i)
    return res

def gen_repeat_macs():
    print("Generating repeat MACs")
    res = set()
    for i in range(16):
        for j in range(16):
            mac = (i << 4) + j  # pair like BB or AB or F3, etc.
            mac = (((((((((mac << 8) + mac) << 8) + mac << 8) + mac) << 8) + mac) << 8) + mac)
            res.add(mac)
    return res

def gen_headers():
    # 5-tuple: src ip addr, src port, dst ip addr, dst port, IP protocol
    # 32 bit, 16 bit, 32 bit, 16 bit, 8 bit. Respectively.
    
    pass

def write_to_file(values, filename):
    with open(filename, 'w') as file:
        file.write('\n'.join(map(str, values)))

    print("Successfully created", filename, "with", len(values), "values")
    return

if __name__ == "__main__":
    print("Executing...")
    # rand_ips = gen_rand_ips(10**6)
    # close_ips = gen_close_ips(2**17)
    # edge_ips = gen_edge_ips(2**17)
    # repeat_ips = gen_repeat_ips()
    # write_to_file(rand_ips, "random_ips.txt")
    # write_to_file(close_ips, "close_ips.txt")
    # write_to_file(edge_ips, "edge_ips.txt")
    # write_to_file(repeat_ips, "repeat_ips.txt")

    rand_macs = gen_rand_macs(10**6)
    close_macs = gen_close_macs(2**18)
    edge_macs = gen_edge_macs(2**18)
    repeat_macs = gen_repeat_macs()
    write_to_file(rand_macs, "random_macs.txt")
    write_to_file(close_macs, "close_macs.txt")
    write_to_file(edge_macs, "edge_macs.txt")
    write_to_file(repeat_macs, "repeat_macs.txt")
