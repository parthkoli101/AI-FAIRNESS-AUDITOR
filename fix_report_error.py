"""
Patches app.py to:
1. Add traceback print inside the generate_ai_report exception handler so Flask terminal shows the real error.
2. Add traceback print inside the run_appeal_engine exception handler.
"""
content = open('app.py', encoding='utf-8').read()

# Patch 1: generate_ai_report exception → add traceback print
old1 = (
    '    except Exception as e:\n'
    '        return {"error": str(e)}\n'
    '\n'
    '# \u2500\u2500\u2500 What-If Simulations'
)
new1 = (
    '    except Exception as e:\n'
    '        import traceback; traceback.print_exc()\n'
    '        return {"error": str(e)}\n'
    '\n'
    '# \u2500\u2500\u2500 What-If Simulations'
)

if old1 in content:
    content = content.replace(old1, new1, 1)
    print("Patch 1 applied: added traceback to generate_ai_report")
else:
    print("Patch 1 NOT applied — searching around line 515...")
    lines = content.split('\n')
    for i, l in enumerate(lines):
        if 'return {"error": str(e)}' in l and 510 < i < 525:
            lines.insert(i, '        import traceback; traceback.print_exc()')
            print(f"  Inserted traceback at line {i+1}")
            content = '\n'.join(lines)
            break

open('app.py', 'w', encoding='utf-8').write(content)

# Verify
c2 = open('app.py', encoding='utf-8').read()
print("traceback.print_exc occurrences:", c2.count('traceback.print_exc()'))
print("Done. Restart Flask to see the real error in the terminal.")
