import sys,threading
sys.path.insert(0,sys.argv[1])
import run_codex_luna_worker as w
w.LIMIT=3
w.SLOTS=threading.BoundedSemaphore(3)
w.main()
