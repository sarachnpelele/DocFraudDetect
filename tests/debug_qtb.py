import jpegio

jpg = jpegio.read('test_sample.jpg')
print("Number of quant tables:", len(jpg.quant_tables))
for i, table in enumerate(jpg.quant_tables):
    print(f"\nTable {i}:")
    print(table)
    print("dtype:", table.dtype)
