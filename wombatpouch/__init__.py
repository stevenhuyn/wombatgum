from glovers import sum_as_string, printer, Foo, Glover

def main():
  print("yo")
  print(sum_as_string(2, 2))
  printer()
  print(Foo)
  glover = Glover()
  print(glover.similar("dog", "cat"))

if __name__ == "__main__":
  main()