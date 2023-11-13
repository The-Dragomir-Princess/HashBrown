from scipy import stats

NUM_BUCKETS = 65536
BUCKET_SIZE = (2 ** 32) // NUM_BUCKETS
NUM_HASH_FN = 4
NUM_ITER = 5
SIZE = 209715
CAPACITY = SIZE * NUM_ITER // NUM_BUCKETS * 3

random_fmt = "build/random-{size}-{iter}"
contiguous_fmt = "build/contiguous-{size}-{iter}"

names = ["CityHash", "FarmHash", "Murmurhash", "xxHash"]
buckets = [[0] * NUM_BUCKETS for _ in range (NUM_HASH_FN)]
histogram = [[0] * CAPACITY for _ in range(NUM_HASH_FN)]
expected = [SIZE / NUM_BUCKETS * NUM_ITER] * NUM_BUCKETS

if __name__ == "__main__":
  # with open("build/random-chisquare", "w") as fout:
  #   for i in range(NUM_ITER):
  #     with open(random_fmt.format(size = SIZE, iter = i)) as fin:
  #       fin.readline()  # skip header
  #       for _ in range(SIZE):
  #         hashes = fin.readline().split(";")
  #         for j in range(NUM_HASH_FN):
  #           hash = int(hashes[j], 16)
  #           buckets[j][hash // BUCKET_SIZE] += 1
  #       # print(fin.readline())
  #   for i in range(NUM_HASH_FN):
  #     res = stats.chisquare(buckets[i], expected)
  #     print(res)
  #     fout.write(f"{names[i]}:{res.statistic}:{res.pvalue};\n")

  # buckets = [[0] * NUM_BUCKETS for _ in range (NUM_HASH_FN)]
  # with open("build/contiguous-chisquare", "w") as fout:
  #   for i in range(NUM_ITER):
  #     with open(contiguous_fmt.format(size = SIZE, iter = i)) as fin:
  #       fin.readline()  # skip header
  #       for _ in range(SIZE):
  #         hashes = fin.readline().split(";")
  #         for j in range(NUM_HASH_FN):
  #           hash = int(hashes[j], 16)
  #           buckets[j][hash // BUCKET_SIZE] += 1
  #   for i in range(NUM_HASH_FN):
  #     res = stats.chisquare(buckets[i], expected)
  #     print(res)
  #     fout.write(f"{names[i]}:{res.statistic}:{res.pvalue};\n")

  buckets = [[0] * NUM_BUCKETS for _ in range (NUM_HASH_FN)]
  with open("build/random-distribution", "w") as fout:
    for i in range(NUM_ITER):
      with open(random_fmt.format(size = SIZE, iter = i)) as fin:
        fin.readline()  # skip header
        for _ in range(SIZE):
          hashes = fin.readline().split(";")
          for j in range(NUM_HASH_FN):
            hash = int(hashes[j], 16)
            buckets[j][hash // BUCKET_SIZE] += 1
    for i in range(NUM_HASH_FN):
      for j in range(NUM_BUCKETS):
        histogram[i][min(buckets[i][j], CAPACITY - 1)] += 1
    for i in range(NUM_HASH_FN):
      fout.write(f"{names[i]}:\n")
      for j in range(CAPACITY):
        fout.write(f"{j}:{histogram[i][j]} ")
      fout.write("\n")
  
  buckets = [[0] * NUM_BUCKETS for _ in range (NUM_HASH_FN)]
  with open("build/contiguous-distribution", "w") as fout:
    for i in range(NUM_ITER):
      with open(contiguous_fmt.format(size = SIZE, iter = i)) as fin:
        fin.readline()  # skip header
        for _ in range(SIZE):
          hashes = fin.readline().split(";")
          for j in range(NUM_HASH_FN):
            hash = int(hashes[j], 16)
            buckets[j][hash // BUCKET_SIZE] += 1
    for i in range(NUM_HASH_FN):
      for j in range(NUM_BUCKETS):
        histogram[i][min(buckets[i][j], CAPACITY - 1)] += 1
    for i in range(NUM_HASH_FN):
      fout.write(f"{names[i]}:\n")
      for j in range(CAPACITY):
        fout.write(f"{j}:{histogram[i][j]} ")
      fout.write("\n")
  
  buckets = [0] * NUM_BUCKETS
  histogram = [0] * CAPACITY
  with open("build/random-2left-distribution", "w") as fout:
    for i in range(NUM_ITER):
      with open(random_fmt.format(size = SIZE, iter = i)) as fin:
        fin.readline()  # skip header
        for _ in range(SIZE):
          hashes = fin.readline().split(";")
          bucket1 = int(hashes[1], 16) // BUCKET_SIZE
          bucket2 = int(hashes[3], 16) // BUCKET_SIZE
          if buckets[bucket1] <= buckets[bucket2]:
            buckets[bucket1] += 1
          else:
            buckets[bucket2] += 1
    for i in range(NUM_BUCKETS):
      histogram[min(buckets[i], CAPACITY - 1)] += 1
    fout.write(f"{names[1]}+{names[3]}:\n")
    for i in range(CAPACITY):
      fout.write(f"{i}:{histogram[i]} ")
    fout.write("\n")
  
  buckets = [0] * NUM_BUCKETS
  histogram = [0] * CAPACITY
  with open("build/contiguous-2left-distribution", "w") as fout:
    for i in range(NUM_ITER):
      with open(contiguous_fmt.format(size = SIZE, iter = i)) as fin:
        fin.readline()  # skip header
        for _ in range(SIZE):
          hashes = fin.readline().split(";")
          bucket1 = int(hashes[1], 16) // BUCKET_SIZE
          bucket2 = int(hashes[3], 16) // BUCKET_SIZE
          if buckets[bucket1] <= buckets[bucket2]:
            buckets[bucket1] += 1
          else:
            buckets[bucket2] += 1
    for i in range(NUM_BUCKETS):
      histogram[min(buckets[i], CAPACITY - 1)] += 1
    fout.write(f"{names[1]}+{names[3]}:\n")
    for i in range(CAPACITY):
      fout.write(f"{i}:{histogram[i]} ")
    fout.write("\n")

  buckets = [0] * NUM_BUCKETS
  histogram = [0] * CAPACITY
  with open("build/random-4left-distribution", "w") as fout:
    for i in range(NUM_ITER):
      with open(random_fmt.format(size = SIZE, iter = i)) as fin:
        fin.readline()  # skip header
        for _ in range(SIZE):
          hashes = fin.readline().split(";")
          target = int(hashes[0], 16) // BUCKET_SIZE
          for j in range(1, NUM_HASH_FN):
            hash = int(hashes[j], 16)
            if buckets[target] > buckets[hash // BUCKET_SIZE]:
              target = hash // BUCKET_SIZE
          buckets[target] += 1
    for i in range(NUM_BUCKETS):
      histogram[min(buckets[i], CAPACITY - 1)] += 1
    fout.write(f"{names[0]}+{names[1]}+{names[2]}+{names[3]}:\n")
    for i in range(CAPACITY):
      fout.write(f"{i}:{histogram[i]} ")
    fout.write("\n")

  buckets = [0] * NUM_BUCKETS
  histogram = [0] * CAPACITY
  with open("build/contiguous-4left-distribution", "w") as fout:
    for i in range(NUM_ITER):
      with open(contiguous_fmt.format(size = SIZE, iter = i)) as fin:
        fin.readline()  # skip header
        for _ in range(SIZE):
          hashes = fin.readline().split(";")
          target = int(hashes[0], 16) // BUCKET_SIZE
          for j in range(1, NUM_HASH_FN):
            hash = int(hashes[j], 16)
            if buckets[target] > buckets[hash // BUCKET_SIZE]:
              target = hash // BUCKET_SIZE
          buckets[target] += 1
    for i in range(NUM_BUCKETS):
      histogram[min(buckets[i], CAPACITY - 1)] += 1
    fout.write(f"{names[0]}+{names[1]}+{names[2]}+{names[3]}:\n")
    for i in range(CAPACITY):
      fout.write(f"{i}:{histogram[i]} ")
    fout.write("\n")
