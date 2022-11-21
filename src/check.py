import numpy

def find_closest_match(test_str, list2check):
  scores = {}
  test_str = test_str.lower()
  for ii in list2check:
      ii = ii.lower()
      cnt = 0
      if len(test_str)<=len(ii):
          str1, str2 = test_str, ii
      else:
          str1, str2 = ii, test_str
      for jj in range(len(str1)):
          cnt += 1 if str1[jj]==str2[jj] else 0
      scores[ii] = cnt
  scores_values        = numpy.array(list(scores.values()))
  closest_match_idx    = numpy.argsort(scores_values, axis=0, kind='quicksort')[-1]
  closest_match        = numpy.array(list(scores.keys()))[closest_match_idx]
  return closest_match, closest_match_idx, scores_values[closest_match_idx], scores_values

def is_str_in(s,l):
    for i in range(len(l)):
        if l[i].lower() == s.lower():
            return True, i
    return False, -1

#print(find_closest_match("Air France",["Air India","Air Costa","Animal"]))
