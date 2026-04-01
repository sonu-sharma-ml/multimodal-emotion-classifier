import pickle
import pickletools

f = open('fer_complete_package_20260401_141226.pkl', 'rb')
print('Pickle protocol:', pickle.DEFAULT_PROTOCOL)
print('\nFirst 15 opcodes:')
for i, (op, arg, pos) in enumerate(pickletools.genops(f)):
    if i >= 15:
        break
    print(f'{op.name}: {repr(arg)[:100] if arg else "None"}')
f.close()