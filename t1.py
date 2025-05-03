gold = """For the 2nd kilometer, the donation will be $10 * 2 = $<<10*2=20>>20.\nFor the 3rd kilometer, the donation will be $20 * 2 = $<<20*2=40>>40.\nFor the 4th kilometer, the donation will be $40 * 2 = $<<40*2=80>>80.\nFor the final kilometer, the donation will be $80 * 2 = $<<80*2=160>>160.\nFor the entire race the donation will be $10 + $20 + $40 + $80 + $160 = $<<10+20+40+80+160=310>>310.\n#### 310', 'For the 2nd kilometer, the donation will be $10 * 2 = $<<10*2=20>>20.\nFor the 3rd kilometer, the donation will be $20 * 2 = $<<20*2=40>>40.\nFor the 4th kilometer, the donation will be $40 * 2 = $<<40*2=80>>80.\nFor the final kilometer, the donation will be $80 * 2 = $<<80*2=160>>160.\nFor the entire race the donation will be $10 + $20 + $40 + $80 + $160 = $<<10+20+40+80+160=310>>310.\n#### 310', 'For the 2nd kilometer, the donation will be $10 * 2 = $<<10*2=20>>20.\nFor the 3rd kilometer, the donation will be $20 * 2 = $<<20*2=40>>40.\nFor the 4th kilometer, the donation will be $40 * 2 = $<<40*2=80>>80.\nFor the final kilometer, the donation will be $80 * 2 = $<<80*2=160>>160.\nFor the entire race the donation will be $10 + $20 + $40 + $80 + $160 = $<<10+20+40+80+160=310>>310.\n#### 310', 'For the 2nd kilometer, the donation will be $10 * 2 = $<<10*2=20>>20.\nFor the 3rd kilometer, the donation will be $20 * 2 = $<<20*2=40>>40.\nFor the 4th kilometer, the donation will be $40 * 2 = $<<40*2=80>>80.\nFor the final kilometer, the donation will be $80 * 2 = $<<80*2=160>>160.\nFor the entire race the donation will be $10 + $20 + $40 + $80 + $160 = $<<10+20+40+80+160=310>>310.\n#### 310"""

resp = """
Okay, so I need to calculate Suzanne's parents' donation based on doubling each successively. First, the donation for the first kilometer is $10.\n- For the second kilometer: $10 * 2 = $20.\n- For the third kilometer: $20 * 2 = $40.\n- For the fourth kilometer: $40 * 2 = $80.\n- For the fifth kilometer: $80 * 2 = $160.\n\nNow, sum these amounts to find the total donation:\n\n$10 + $20 + $40 + $80 + $160 = $310.\n\nTherefore, Suzanne's parents will donate $310 if she finishes the 5-kilometer race.
"""

from math_verify import parse, verify

print(verify(gold, resp))

print(parse(gold))
print(parse(resp))
print(verify(parse(gold), parse(resp)))
