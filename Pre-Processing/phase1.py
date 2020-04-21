
def phase1(sentence, corrections):
    sections = []
    for c in corrections:
        sections = c.split('|||') #0 is range, 1 is correction-type, 2 is correction
        sections[0] = [sections[0].split(' ')[1], sections[0].split(' ')[2]] #0,0 is first index, 0,1 is second index
        print(sections)

    return