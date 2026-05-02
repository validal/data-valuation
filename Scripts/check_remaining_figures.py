#!/usr/bin/env python3
import os
root = os.path.join('Plots', 'fine_grained')
matches = []
for dp, _, fs in os.walk(root):
    for f in fs:
        if f.startswith('Figure_'):
            matches.append(os.path.join(dp, f))
print(len(matches))
for m in matches:
    print(m)
