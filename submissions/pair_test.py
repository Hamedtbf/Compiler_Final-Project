from plagiarism.compare import compare_two_files, compare_hierarchies
from plagiarism.config import load_config

for i in range(1, 12):
    print()
    config = load_config()
    result = compare_two_files(f"pair{i}_a.py", f"pair{i}_b.py",config)
    print(f"Comparison results{i}: ")
    print(f"  token similarity: {result['token']:.4f}")
    print(f"  AST similarity:   {result['ast']:.4f}")
    print(f"  CFG similarity:   {result['cfg']:.4f}")
    print(f"  FINAL score:      {result['final']:.4f}")
    print()

    report = compare_hierarchies(f"pair{i}_a.py", f"pair{i}_b.py", config)
    # print a readable hierarchical summary
    print(f"Hierarchical comparison{i}: ")
    print("Functions:")
    for fn, info in report.get("functions", {}).items():
        print(
            f"  {fn}: final={info['final']:.4f} (token={info['token']:.4f}, ast={info['ast']:.4f}, cfg={info['cfg']})")
    print("Classes:")
    for cn, cinfo in report.get("classes", {}).items():
        print(
            f"  {cn}: final={cinfo['final']:.4f} (token={cinfo['token']:.4f}, ast={cinfo['ast']:.4f}, cfg={cinfo['cfg']})")
        for m, minfo in cinfo.get("methods", {}).items():
            print(
                f"    {m}: final={minfo['final']:.4f} (token={minfo['token']:.4f}, ast={minfo['ast']:.4f}, cfg={minfo['cfg']})")
    print("Variables:")
    for v, vinfo in report.get("variables", {}).items():
        print(f"  {v}: final={vinfo['final']:.4f} (token={vinfo['token']:.4f}, ast={vinfo['ast']:.4f})")
    print("-----------------------------------------------------------------------")