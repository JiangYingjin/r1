gold = """
First convert Haley's shopping time to minutes: 2 hours * 60 minutes/hour = <<2*60=120>>120 minutes
Then convert Haley's setup time to minutes: .5 hours * 60 minutes/hour = <<.5*60=30>>30 minutes
Then find the total time she spends making a snack: 3 * 30 minutes = <<3*30=90>>90 minutes
Then add the time she spends on each activity to find the total time: 120 minutes + 30 minutes + 90 minutes = <<120+30+90=240>>240 minutes
Then divide the time she spends watching the comet by the total time and multiply by 100% to express the answer as a percentage: 20 minutes / 240 minutes = 8.333...%, which rounds down to 8%
#### 8"""

gold = "8%%"

resp = """
$$
\\boxed{8}
$$
"""

resp = "\\boxed{8\%}"

from math_verify import verify, parse

parsed_gold = parse(gold)
parsed_resp = parse(resp)
print(f"gold: {parsed_gold}")
print(f"resp: {parsed_resp}")
print(verify(parsed_gold, parsed_resp))
