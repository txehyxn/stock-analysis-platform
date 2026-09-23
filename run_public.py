import os
import sys
import time
import threading
import signal

# Windows 콘솔 UTF-8 출력 보장
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import uvicorn
from pycloudflared import try_cloudflare

# 상위 디렉토리 모듈 참조
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from web.app import app

PORT = 8000
HOST = "127.0.0.1"

class ServerThread(threading.Thread):
    def __init__(self, app, host, port):
        super().__init__()
        config = uvicorn.Config(app, host=host, port=port, log_level="warning")
        self.server = uvicorn.Server(config)
        self.daemon = True

    def run(self):
        self.server.run()

    def stop(self):
        self.server.should_exit = True

def main():
    print("=" * 70)
    print("[*] FastAPI 웹 서버 기동 중: http://127.0.0.1:8000 ...")
    server_thread = ServerThread(app, HOST, PORT)
    server_thread.start()

    # 서버 기동 대기
    time.sleep(1.5)

    print("[*] Cloudflare 공인 HTTPS 터널 생성 중 (pycloudflared)...")
    tunnel_urls = None
    try:
        tunnel_urls = try_cloudflare(port=PORT, verbose=False)
        public_url = str(tunnel_urls.tunnel).strip()

        print("\n" + "=" * 70)
        print("  [SUCCESS] Cloudflare 공인 HTTPS 터널이 성공적으로 활성화되었습니다!")
        print("=" * 70)
        print(f"\n  >> 스마트폰 접속 HTTPS 주소:  {public_url}\n")
        print(f"  >> 로컬 접속 주소          :  http://localhost:{PORT}")
        print("=" * 70)
        print("\n[안내] 스마트폰(Safari, Chrome)에서 위 HTTPS 링크로 접속하여")
        print("       1D-CNN 예측 차트와 종목 대시보드를 바로 테스트할 수 있습니다.")
        print("[종료] 터미널에서 Ctrl + C 를 누르면 터널과 서버가 안전하게 종료됩니다.\n")
        sys.stdout.flush()

        # 메인 루프 유지
        while server_thread.is_alive():
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n[*] Cloudflare 터널 및 FastAPI 서버를 안전하게 종료합니다...")
    except Exception as e:
        print(f"[-] 터널 실행 중 오류 발생: {e}")
    finally:
        if tunnel_urls and hasattr(tunnel_urls, "process"):
            try:
                tunnel_urls.process.terminate()
            except Exception:
                pass
        try:
            try_cloudflare.terminate(PORT)
        except Exception:
            pass
        server_thread.stop()
        print("[v] 모든 프로세스가 정상 종료되었습니다.")

if __name__ == "__main__":
    main()
