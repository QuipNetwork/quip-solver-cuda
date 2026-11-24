# Ghi chú tối ưu miner Otsuka (chain 120, family twisted-ice-dimer) — ĐỌC TRƯỚC KHI TỐI ƯU

Cập nhật: 2026-11-14 ~06:10 giờ VN. Bản đang chạy: opt26 (sàng q8/q8/q4 + prior + đua dòng dõi + dòng dõi phụ). Giữ lại để quay về: opt22, opt17, opt5, opt8, rc6 (opt19 chỉ còn trong sao lưu). Bản sao lưu: /home/vanthoi/lam-viec-tam-backup/best-chain120/
Quy tắc: trước khi thử hướng mới, tìm hướng đó trong file này. Sau mỗi thử nghiệm, ghi kết quả (dù thất bại) vào bảng.

## Bối cảnh cần nhớ
- Ngưỡng chain là bộ điều khiển quanh -5077 ± 1 (không leo thang theo đối thủ). Đối thủ (0xcd45a1, 0x5abae3, một GPU) thắng ~mỗi 10 phút với nonce -5075…-5080.
- Chất lượng mỗi attempt của họ gần ta nhưng họ có nhiều attempt hơn khoảng 5–8 lần (ước tính). Mục tiêu thực: tăng số nghiệm chất lượng cao mỗi giây GPU.
- Độ nhạy đuôi: +1 đơn vị năng lượng trung bình ≈ nhân tốc độ chạm ngưỡng ~2,5 lần (z≈3, σ≈3,3). Chỉ đáng đổi chất lượng lấy thời gian nếu +>0,7 đơn vị mỗi lần gấp đôi thời gian.
- Harness (CU4_*) dùng đồ thị Zephyr 4800 node, chain thật 4575 node: chỉ so sánh TƯƠNG ĐỐI, độ sâu bão hòa khác nhau. Luôn xác nhận bằng số liệu attempt thật (dashboard attempts/<qblock>/attempts.jsonl, lọc theo ts sau lần deploy).
- Cửa sổ sàng cần 15–20 phút để ổn định sau mỗi lần khởi động lại; tránh restart thường xuyên. Mỗi lần dừng miner để đo là mất cơ hội thắng.

## TIÊU CHÍ TRIỂN KHAI MỚI (người dùng 14/11 chiều): chỉ triển khai khi nonce đạt SÂU HƠN -5085 (siêu thay thế mức -5079,9 bên dưới)

## TIÊU CHÍ TRIỂN KHAI (người dùng 11:10 14/11)
- Một phiên bản chỉ đáng triển khai khi nonce tốt nhất mỗi round của nó ≥ (sâu hơn hoặc bằng) mức TRUNG BÌNH nonce thắng của đối thủ. Chỉ chạm -5077 gần như không thể thắng.
- KPI (14 round gần nhất đến 11:21): nonce thắng của đối thủ -5079…-5081, trung bình ≈ -5079,9 (cả 14 giờ qua: 73 khối của đối thủ, trung vị -5078; mức đầu ngày nông hơn vì ngưỡng thấp hơn). Nonce tốt nhất MỖI ROUND của ta: -5057,8…-5075,3, trung bình ≈ -5067,6; chỉ 3/14 round đạt ≥ -5074. Khoảng cách trung bình ≈ 12; ở round tốt nhất ≈ 4,6.
- Hệ quả: các tinh chỉnh nhỏ (+1…+2) không đủ; cần thay đổi bậc để nâng nonce tốt nhất mỗi round thêm ~12, hoặc chấp nhận xác suất thắng thấp (ta vẫn thắng 2 khối lúc 05:50 và 07:39 với opt17: nonce -5079).
- Đo KPI: dựng lại bằng script trong ghi chú này (log coordinator 'stopping miners' + attempts.jsonl theo khoảng thời gian round).

## MỤC TIÊU NGẮN HẠN (người dùng 12:10 14/11): làm cho nonce đạt -5085 trước mọi quyết định khác
- Quy tắc nộp: coordinator nộp bất kỳ nonce nào đạt ngưỡng; độ sâu tối đa chỉ là 'càng sâu càng tốt'.
- Quan sát của người dùng: attempt của đối thủ dài 1–7 phút và dừng khi chạm ngưỡng (thời gian thành công kiểu hình học; trung bình ~3 phút ⇒ xác suất thành công ~30% mỗi phút/dòng dõi). Họ tiếp tục trên cùng nonce đến khi thành công; ta chạy một lượt cố định rồi bỏ.
- Mô phỏng sự kiện (ceiling/sim_*.py; hiệu chuẩn: live -5079 ↔ mô hình -18,5; live -5085 ↔ -26,9): số lần chạm -5085 mỗi 24 h ở tốc độ x1: opt22 0; q64/32/16: 0,17; thêm 6 dòng dõi phụ nếu deep<=-20: 0,08; ở x4: 0,42–0,58; ở x10: 0,4–1,7. Tức là -5085 cần cả sàng cực đoan lẫn thông lượng gấp ~4–10 (kernel nhanh hơn); với phần mềm hiện tại kỳ vọng ≈ vài ngày mới có một lần.
- Giới hạn giao thức: tầng phiên xóa bản ghi chờ của job ngay sau kết quả đầu tiên ⇒ KHÔNG gửi được kết quả sớm giữa chừng cho cùng một job. Chỉ có thể kéo dài job (tiếp tục dòng dõi mới trên cùng instance) rồi trả một lần.
- opt23 (12:18): (1) cửa sổ sàng khởi tạo sẵn bằng ngưỡng tham chiếu đo từ dữ liệu live (A -5008, B -5059,5, C -5072,5; QS_PRIOR) ⇒ không còn 15–20 phút làm nóng; (2) attempt độ dài biến thiên: sau run sâu, nếu best ∈ (-5081, -5074] chạy thêm tối đa 6 dòng dõi (seed khác) trên cùng instance cho tới khi best ≤ -5081 (QS_EXIT), round bị hủy hoặc hết giới hạn (QS_PROMISE, QS_FOLLOW).

## opt24 (12:24 14/11): đua dòng dõi (lineage racing) + dòng dõi phụ — hướng rút ra từ gợi ý của người dùng
- Cơ sở đo: năng lượng nhiệt tốt nhất của quần thể ở mốc 25% lịch ủ tương quan 0,81 với kết quả cuối (50%: 0,95; ceiling/trace.tsv). Vậy chất lượng một dòng dõi quyết định sớm; chạy hết 100% mỗi dòng dõi là lãng phí.
- Cài đặt: ở mốc 25% của run sâu (anneals>=2048) so năng lượng nhiệt tốt nhất với cửa sổ 64 run gần nhất (cần >=16 mẫu); bỏ nếu không thuộc nửa tốt nhất (QC_RACE_AT=0,25; QC_RACE_KEEP=0,5; QC_RACE=0 để tắt — HARNESS PHẢI đặt QC_RACE=0). Dòng dõi bị bỏ trả lại nghiệm của tầng C. Kết hợp dòng dõi phụ (QS_FOLLOW=12) cho attempt dài biến thiên giống đối thủ.
- Kỳ vọng: ~1,6x số dòng dõi mỗi phút GPU và dòng dõi giữ lại tốt hơn trung bình (top 50% theo ρ=0,81) ⇒ tần suất chạm ngưỡng sâu tăng ~2–3x. CHƯA đo live: cần đếm thời gian job (aborted ~14 s vs đầy đủ ~55 s) và số lần chạm ≤ -5075.

## BÀI HỌC opt22–24 và opt26 (12:40 14/11)
- opt22/23/24 dùng sàng nghiêm q16/q16/q8: >90% thời gian GPU ở tầng A, gần như KHÔNG có run sâu (0 deep trong 26 phút; 3 job C trong 2619 job) ⇒ nonce tốt nhất mỗi round chỉ là mức tầng sàng (-5060…-5065). Mô hình mô phỏng sai vì chi phí job A thực là ~0,19 s (không phải 0,13 s) và tỉ lệ qua tầng thực nhỏ hơn danh nghĩa. KHÔNG dùng q16/q16/q8 nữa.
- opt26: stages 256:1024:8;256:8192:8;512:16384:4 (như opt17) + prior cửa sổ -5004;-5052,5;-5064 (QS_PRIOR; hiệu chỉnh từ phân bố live: A thr ≈ -5004 qua ~12%, B ≈ -5052,5, C512 ≈ -5064 vì C512 trung vị -5059,6) + đua dòng dõi ở 25% + dòng dõi phụ (QS_FOLLOW=12, QS_EXIT=-5081, QS_PROMISE=-5074). Luồng sau 5 phút đầu: A 567, B 62, C 7, deep 1 (54 s, -5071,1).
- Khi đổi cấu hình sàng PHẢI kiểm tra số run sâu mỗi 10 phút trên dữ liệu live trước khi giữ lại.

## ƯU TIÊN (người dùng 12:50 14/11)
- Chưa làm gì về 'nộp sớm/tranh thứ tự nộp' cho đến khi có khối TRÙNG nonce với đối thủ (cùng block). Hiện chỉ tập trung nâng độ sâu nonce đạt ngưỡng.

## Kết quả thực tế
- 2026-11-14 12:53:36 VN: opt26 THẮNG khối với nonce -5082 (job dài 103 s = 2 dòng dõi trên cùng instance, best -5081,3 → dòng dõi phụ HOẠT ĐỘNG). Trong 23 phút đầu: 5 job sâu (4 x 54 s: -5071,1…-5074,0; 1 job 103 s: -5081,3). Hai lần follow-up khác (~300 s mỗi lần) bị hủy khi đối thủ thắng (12:51:37, 13:00:56): khoảng trống ~300–340 s trong attempts.jsonl là follow-up đang chạy/bị hủy, không phải GPU rảnh.
- 2026-11-14 05:50 VN: opt17 THẮNG khối với nonce -5079,08 (job sâu tốt nhất; trung vị job sâu -5073,9, 8 mẫu, ~120 s/job). Trước đó thắng cuối cùng lúc 00:30 (opt8).

- 2026-11-14 06:33 VN: opt17 13 job sâu: tốt nhất -5079,1; trung vị -5074,2; 3 job <= -5075. Đối thủ 4 khối/65 phút (nhịp giãn ~19 phút).
- Cân nhắc 06:33 (chưa làm, không thử nghiệm): racing trong deep ít giá trị vì tầng C đã sàng bằng run đủ lịch ủ; warm start từ tầng C gần giống ủ lặp (đã thất bại) nên rủi ro.
- 2026-11-14 07:03 VN: đối thủ 0x5abae3 sâu hơn (-5081, -5083); ta 0 khối/65 phút. opt17 20 job sâu: tốt nhất -5079,1; trung vị -5073,5; 4 job <= -5075. Các nút tinh chỉnh đã hòa vốn; cần hướng thuật toán mới (xem mục chưa thử).
- 2026-11-14 ~07:40 VN: đối thủ chạy attempt 1–3 phút và nộp nonce sâu nhất trước khi bị hủy (theo người dùng quan sát); đã thử PT làm kiểu 'anytime' — kém xa PA. Mục tiêu người dùng đặt: nonce -5085.
- 2026-11-14 07:39 VN: ta THẮNG thêm 1 khối (-5079) khi còn chạy opt17. 08:10: opt19 vừa khởi động (08:06), GPU 99,1% thời gian thực; chờ job sâu để đo.

## TRẦN của thuật toán hiện tại (đo 08:25–08:55 VN 14/11, harness, 4096→256 @0,35, PA 32/0,3, beta 1.2–8)
| Instance | 32768 sweeps | 131072 sweeps | 524288 sweeps |
|---|---|---|---|
| 3 (tốt) | -5682,5 | -5690,4 / -5692,0 (2 seed) | |
| 0 (tốt nhất) | | -5693,8 | **-5691,3** (không cải thiện, thậm chí kém hơn) |
| 7 (kém, -5652 ở 16384 x 256) | | -5686,7 | |
- Kết luận: với instance tốt nhất đã biết, độ sâu bão hòa quanh 131072 sweeps (524288 không tốt hơn); trần thuật toán ≈ -5694 (harness) cho instance này. Quần thể 8192 không hơn 4096 (xem bảng chung). Tăng sweeps/quần thể/reheat/PT/gộp đều không phá được trần.
- Quy đổi chain thật (mức cải thiện harness ≈ 1,4x live): instance điển hình ở độ sâu hiện tại ≈ -5075 live ↔ harness -5688; instance tốt nhất -5694 ↔ khoảng -5079…-5080 live. Muốn -5085 live cần ≈ -5702 harness: ngoài trần của thuật toán này; chỉ đạt nếu tìm instance tốt hơn nữa (kéo đuôi σ_inst≈6 harness) hoặc có thuật toán mới.
- Hướng còn lại khả dĩ: (1) tăng kích thước pool instance (kéo đuôi σ_inst, +0,5…+1 live cho mỗi lần gấp đôi pool với chi phí ~+18% chu kỳ); (2) best-of-N seed cho top instance (sd seed ≈ 1,1 harness, +1,5 cho N=4); (3) thuật toán động lực học mới (worm/loop có hướng) — chưa làm.

## Chiến dịch trần giai đoạn 1 (65536 sweeps, instance 0; ceiling/stage1.tsv)
- Gốc 4096→256 @0,35: -5688,7/-5688,1/-5687,2 (3 seed, sd 0,7). Biến thể (seed 11): pop2048 -5693,4 (48 s), shrink@0,25 -5692,1, shrink_to1024 -5691,9, beta 0,8–8 -5691,6, frac0,5 -5691,4, beta 1,2–10 -5691,4, pop8192 -5691,1, shrink_to512 -5690,0, every16 -5690,2, beta 1,2–6 -5689,9, every64 -5689,8, shrink@0,5 -5689,3, beta 1–8 -5689,3, frac0,15 -5688,4, shrink_to128 -5687,1, beta 1,5–8 -5685,3.
- Lưu ý quan trọng (người dùng nhắc 09:29): -5085 chỉ là trần của thuật toán HIỆN TẠI; mỗi khi đổi thuật toán/tham số cần đo lại trần trước khi kết luận. Trần -5694 ở mục trên được ĐO VỚI pop 4096; pop 2048 cho -5693,4 ở 65536 sweeps (nhanh hơn một nửa) nên trần có thể nới.

## Kết luận về viết lại thuật toán (11:55 14/11)
- Cảnh quan năng lượng của instance có cấu trúc 'bồn': mỗi lượt PA rơi vào một bồn sâu; năng lượng các bồn lệch nhau vài đơn vị (seed khác nhau trên cùng instance: -5691,0 vs -5686,3; sd seed ≈ 2,5 harness), và hai bồn khác nhau ~40% nút. Mọi cải thiện cục bộ (cụm, worm, DP chính xác 250–400 nút, tái tổ hợp) không vượt được bồn.
- Do đó đòn bẩy duy nhất còn lại là SỐ LƯỢNG bồn/instance được lấy mẫu mỗi giây GPU (best-of-N seed, chọn instance) — đã được tối ưu bằng sàng/PA/shrink; tăng tốc kernel thêm 2x chỉ ≈ +1…+2. Không có thuật toán viết lại nào đo được là cho bậc +12.

## Phân tích 'vì sao thua đối thủ' (09:50 14/11)
- Đối thủ thắng 5/5 round gần nhất; nonce -5080..-5083; ta tốt nhất mỗi round ≈ -5072..-5077.
- Pair worm upstream không cải thiện nghiệm GPU (gain ~0) ⇒ ta gần trạng thái cơ bản của instance đang giải. Năng lượng cơ bản giữa các instance có sd ≈ 6 (harness) ≈ 4 live ⇒ nonce sâu nhất phụ thuộc vào việc TÌM ĐƯỢC instance có trạng thái cơ bản thấp nhất trong round, tức chất lượng xếp hạng instance và số instance sàng được mỗi giây. Giả thuyết: attempt 7–15 s của đối thủ là các lần sàng.
- Hướng thiết kế: sàng liên tục (successive halving) có tiếp nối trạng thái giữa các mức thay vì khởi động lại từng tầng; đang đo độ tương quan dự báo của năng lượng ở các mốc giữa lượt ủ (ceiling/trace.tsv).

## Thống kê instance (đo 10:13 14/11; ceiling/trace.tsv; 20 instance, 65536 sweeps, pop 2048→256 @0,35)
- Năng lượng cuối giữa các instance: sd 5,3; khoảng -5693,4 … -5670,7 (chênh 22) với cùng thuật toán và cùng ngân sách ⇒ nonce sâu nhất chủ yếu do CHỌN được instance tốt.
- Tương quan của năng lượng nhiệt giữa lượt ủ (tốt nhất quần thể) với kết quả cuối: mốc 512: 0,02; 1024: 0,47; 2048: -0,04; 4096: 0,46; 8192: 0,35; 16384 (25%): 0,81; 32768 (50%): 0,95. Mốc giữa lượt kém hiệu quả so với sàng khởi động lại (tầng B 1,2 s ρ≈0,73) ⇒ KHÔNG làm 'racing tiếp nối'.
- Ước lượng trần theo thống kê: best-of-N instance ≈ trung bình + z(N)·5,3 harness; N≈800 (một round với pipeline hiện tại) z≈3,1; N≈10^4 z≈3,7; để -5085 live (≈ +10 harness so với hiện tại) cần z≈5 ⇒ N≈10^6, ngoài tầm. Trần thực tế trong một round ≈ -5080…-5082 live. Nâng N (gộp nhiều instance mỗi lần launch kernel để sàng nhanh 8–16x) chỉ thêm ≈ +1,8 live.
- opt20 (10:14): deep bắt đầu 2048 reads (thu về 256 @0,35, 131072 sweeps) ≈ 55% thời gian run sâu so với opt19 ở chất lượng tương đương.

## Sàng song song (opt21, 10:33 14/11) — CÓ HIỆU QUẢ về thông lượng
- 4 worker thread, mỗi worker một CUDA stream riêng (GpuCluster.kernels), tầng A chỉ 64 reads, B 128 reads, C 512. Nhiều job sàng chạy đồng thời lấp đầy GPU thay vì mỗi job chiếm cả GPU 0,13 s.
- Kết quả 3 phút đầu: GPU 100% / 443–449 W (sát TDP), 7,7 job/s (trước ~3,7 job/s), A median 0,171 s/job (đồng thời). Chưa đo chất lượng job sâu: cần ≥10 job sâu so với opt20/opt19 (trung vị -5072).
- Cấu hình: QS_WORKERS (mặc định 4), QS_STAGES "64:1024:8;128:8192:8;512:16384:4".

## opt21 THẤT BẠI (sàng song song 4 worker, A 64 reads, B 128 reads) — hoàn tác 11:06
- Live: deep median -5058,9 (3 mẫu) so với opt19 -5072,8 (14 mẫu); các round -506x. Người dùng đề nghị: khi cần đo liên quan GPU hãy DỪNG miner (kết quả chính xác, nhanh hơn, và chạy song song không thắng mà còn lãng phí).
- Đo sạch trên GPU trống (ceiling/rank.tsv, 20 instance x 2 seed, tương quan với deep một seed): A256 0,38 (0,13 s); A64 0,29 (0,10 s); B256 0,61 (0,90 s); B128 0,49 (0,77 s); C512 0,70 (3,45 s); C256 0,61 (1,77 s). Ít reads làm xếp hạng kém.
- Mô phỏng Monte Carlo (ceiling/simulate_pipeline.py, ngân sách 600 s/round, deep 55 s): opt21-like -13,75 so với opt20-like -14,01 (chỉ kém 0,3 ⇒ KHÔNG giải thích -5058; nghi do 4 worker chia GPU làm job sâu bị chậm/hủy). Cấu hình tối ưu theo mô hình: A256/q16 → B256/q16 → C256/q8 (-15,05, +1,0).
- opt22 (11:06): 1 worker tuần tự, stages 256:1024:16;256:8192:16;256:16384:8, deep pop 2048→256 @0,35, 131072 sweeps, keep 16.
- QUY TẮC mới: mọi đo đạc GPU làm khi miner DỪNG, gom thành đợt ngắn.

## Vì sao harness -56xx còn chạy thật -507x (đã giải thích cho người dùng)
- Năng lượng = NỀN cố định (mỗi tile -2,0 đơn vị x số tile, mỗi cặp sắt từ -1,0) + phần "yếu" (cạnh ±62 milli, phần thuật toán tối ưu).
- Harness: 4800 nút, 2116 tile → nền -5432; nghiệm -5687 → phần yếu ≈ -255. Chain thật: 4575 nút, ~1800 tile (ước tính) → nền ≈ -4830; nghiệm -5075 → phần yếu ≈ -245. Chênh -560 là do số tile, không phải chất lượng.
- Mức cải thiện giữa các cấu hình ở harness ≈ 1,4x so với chain thật. CẦN xác nhận số tile thật bằng log ở lần deploy tới.

## Việc dùng GPU (08:06 14/11)
- Trước opt19: tổng thời gian GPU chỉ 86% thời gian thực; job sàng tầng A 0,196 s thay vì 0,13 s (tải/chấm lại/gửi 256 nghiệm). Sau opt19 (chỉ trả 16 read tốt nhất, xếp hạng trên GPU): A 0,132 s, GPU 99% thời gian, 84–98% util, 390–438 W.

## Hướng ĐÃ LÀM và CÓ HIỆU QUẢ (đang nằm trong opt17)
| Hướng | Kết quả | Ghi chú |
|---|---|---|
| Tăng ngân sách (reads/sweeps) so với sàn upstream 64x256 | trung vị -4886 → -5050 | độ sâu bão hòa trên chain thật ~131072 sweeps |
| Dải beta 1.2–10/8 (mặc định upstream kém hơn) | +3…+4 | beta_min >1.5 rất tệ; beta_max 5–12 phẳng |
| Population annealing (PA) trên GPU: mỗi 32 sweeps thay 30–50% read tệ nhất bằng bản sao read tốt | +34 (harness) | lợi ích lớn nhất. Chọn lọc bằng kernel energy + kernel resample 1 lần (không copy từng hàng) |
| Kernel nhanh ~2,6x: find_root volatile + path halving (bỏ atomicAdd đọc), cache xác suất bond theo trọng số, 32 sweep/bậc beta (gộp launch) | wall 2706→~1030 ms (8192 sw x 256 reads) | kết quả bit-giống hệt khi gộp bậc |
| Sàng instance (mỗi salt = một instance khác nhau, sd ~6 harness): chạy ngắn, chỉ đào sâu job trong top 1/q của cửa sổ | +4…+6 live | 3 tầng: 256:1024 q8 → 256:8192 q8 → 512:16384 q4 → deep |
| Quần thể deep 512→2048 reads | +2,2 (512→1024), +1,5 (1024→2048) | 2048 ≈ hòa vốn, KHÔNG tăng thêm |
| Thu nhỏ quần thể sau pha PA (SHRINK): bắt đầu R reads, ở 35% lịch giữ 256 tốt nhất | 4096→256 @0,35: -5687,5 (49 s) vs 2048 đầy đủ -5684,5 (66 s) vs 256 thuần -5678,6 (10 s); 4096 @0,25: -5685,4; @0,15: -5678,7 (quá sớm); 8192 @0,25: -5687,7 (62 s) | CÓ HIỆU QUẢ: quần thể lớn chỉ cần ở giai đoạn nóng/giữa; sau khi sụp, 256 là đủ. Đang ở opt18 (deep 4096→256 @0,35) |
| PA frac 0,3 (thay 0,5) | +3,9 harness (6 seed) | 0,2–0,3 phẳng |

## Hướng ĐÃ THỬ và KHÔNG HIỆU QUẢ (đừng lặp lại)
| Hướng | Kết quả | Vì sao |
|---|---|---|
| quip-cpu-dimer làm miner phụ (CPU 24 luồng) | trung vị -4805, làm GPU chậm 17% | budget sàn, kém xa GPU |
| Thêm luồng CPU / chunk launch (khi sweep mỗi launch=1) | không đổi | không bị giới hạn bởi launch trước khi gộp bậc |
| single_every ≠ 1 | kém hơn (0 → -5538) | cần single-spin mỗi sweep |
| rounds 1/4/8/16 | phẳng (rounds=1 nhanh 6% nhưng PA +11 kém) | |
| tile/dimer weight (Params) | không tác dụng | trọng số lấy từ mảng, hằng số bị bỏ qua |
| Ủ lặp (reheat 2–8 chu kỳ) | kém hơn | mất trạng thái lạnh |
| Hình dạng lịch beta (lũy thừa 0,6–2,5) | phẳng | geometric đủ tốt |
| PA chỉ ở giai đoạn đầu (until 5–75%) | kém | PA cần chạy đến ~35% lịch |
| PA dày hơn (every 8/16) / thưa (128/512/1024) / frac 0,05–0,75 | phẳng ±3 | |
| Boltzmann/Chọn lọc theo xếp hạng sau quench (QC_PA_QUENCH) | phẳng | |
| Đảo (islands) cho PA | không giúp | chất lượng phụ thuộc kích thước quần thể (~log), không phải số dòng dõi |
| Quần thể > 2048 reads | +<1,5 cho x2,3 thời gian | hòa vốn |
| Hệ số nhiệt độ bond cụm λ (0,5–1,5) | phẳng | |
| beta_max 5/6 vs 8 (khi PA) | +1,5 (nhiễu) | |
| Số chu trình frustration (gauge_obstructions) làm bộ dự báo instance | tương quan -0,1 | vô dụng |
| Sàng bằng 1 tầng ngắn (ρ≈0,45) | +4…+5 | đã thay bằng 3 tầng |
| Parallel tempering (64/16/32/8 mức, 1.2–8, 2–6, hoán đổi beta mỗi 32 sweeps, pool nghiệm tốt nhất mọi thời điểm sau quench) | -5622…-5631 so với PA -5683 cùng ngân sách (1024 reads x 32768 sweeps) | kém ~55; cần chạy lâu hơn nhiều mới cân bằng được ở nhiệt độ thấp; swap_acc 0,17–0,73. Bản vá: experiment-parallel-tempering.patch |
| Ủ lặp (reheat 2–3 chu kỳ, hâm lại tới beta 2–4) CÓ PA | -5674 so với -5682 (1 chu kỳ), 512 reads x 32768 sweeps, 3 seed | vẫn kém; lần trước thử không có PA, lần này có PA, cùng kết luận |
| Partition crossover/gộp quần thể (tách thành phần liên thông của tập khác biệt với trạng thái tốt nhất, nhận thành phần làm giảm năng lượng) ở cuối run | gain -0,25 / 0 | quần thể cuối đã sụp thành bản sao gần giống hệt |
| Gộp định kỳ giữa run (mỗi 2048 sweeps từ 10%) rồi nhân bản 30% | cuối -5680,6 so với -5682 | gain giữa run (-10…-51) chỉ là nhiễu nhiệt, tái nhiệt hóa ngay |
| Pair worm của upstream (quip-solver-cpu `ice-pairs-pw/1`: descent tự nhiên + worm DFS có quay lui + crossover + ủ–quench) áp dụng lên 16 nghiệm tốt nhất của GPU | gain -0,12 / 0 (Budget 0,7 s, 16x300 chu kỳ 17 s, 16x3000 chu kỳ 170 s) | nghiệm GPU đã là cực tiểu ổn định đối với mọi loại bước nhảy của họ ⇒ ta đã gần trạng thái cơ bản của chính instance đó; nonce sâu nhất do CHỌN INSTANCE, không do giải sâu hơn. Code: src/pairworm.rs, harness CU4_PW |
| Kernel dimer của upstream (`quip-cpu-dimer`, LOOP64: spin đơn + khối tứ diện chính xác + 64 vòng ice/sweep; 64 reads, 1 luồng CPU) | -5484 (500 sweeps, 2,5 s), -5539 (2000, 10 s), -5570 (8000, 42 s) so với GPU deep -5687 | yếu hơn rất nhiều, tăng chậm (+30 mỗi 4x sweeps); KHÔNG phải nguồn lợi thế của đối thủ |
| Trọng số liên kết mạnh đưa vào bộ cụm (Weights tile/pair 250/500/1000 và tổ hợp) | -5684,0…-5684,7 (3 seed, pop 1024→256, 32768 sweeps) | phẳng; chỉ giá trị cực đoan (1/0: -5639; 50/0: -5650 so với -5652 ở 8192 sweeps) mới đổi |
| Tìm kiếm cận lân cận CHÍNH XÁC trên đồ thị con bề rộng cây thấp (kiểu Hamze–de Freitas/Selby; quy hoạch động loại biến; treewidth ≤6/8/10/12, 60–400 nút; CPU; ceiling/treedp/) áp lên nghiệm tốt nhất của GPU | 0 cải thiện trong 10311/2972/895/283 lần thử (instance 0 -5691,0), 0 trên -5686,0 và -5686,3; chỉ sửa được khác biệt nhỏ của nghiệm hạng 16 trong quần thể về đúng nghiệm tốt nhất | nghiệm GPU tốt nhất là cực tiểu rất sâu trong lân cận khổng lồ; cải thiện CHỈ có thể đến từ bước nhảy cỡ bồn (nghìn nút) |
| Tái tổ hợp chính xác hai nghiệm sâu của hai seed trên cùng instance (-5691,0 và -5686,3; tập khác biệt 1952/4800 nút, giải DP chính xác trên các cửa sổ 250 nút nằm trong tập khác biệt) | 0 cải thiện theo cả hai chiều | hai nghiệm nằm ở hai 'bồn' khác nhau; chênh 4,7 là kết quả của đảo cả miền lớn, không tách được thành miền độc lập nhỏ |
| Worm/loop moves (polish thấp nhiệt) | chỉ 8% worm đóng, độ dài 2 | >50% nút nằm trong cặp sắt từ → siêu đồ thị; cần BFS vòng có hướng — chưa làm |
| Độ sâu >131072 sweeps (live) | opt9 262144 = opt8 131072 | bão hòa |
| threads/block 128–1024, unroll vòng hàng xóm | không đổi | ncu: occupancy 66% (thanh ghi+shared), long_scoreboard 25%, barrier 22% |
| 3 block/SM (int16 parent) | chưa làm | lợi tối đa ~10–15% |

## Ý tưởng CHƯA THỬ (theo thứ tự đáng thử)
1. Racing trong chính run deep: bỏ giữa chừng nếu năng lượng ở checkpoint (vd 4096, 16384 sweeps) không nằm top so với lịch sử cùng checkpoint (không bỏ phí compute như sàng khởi động lại).
2. Warm start: dùng quần thể kết thúc của tầng C làm điểm bắt đầu cho deep (reheat vừa phải) thay vì chạy lại từ nóng.
3. Worm/loop có hướng (BFS vòng trên siêu đồ thị tile/pair) làm bước hoàn thiện thấp nhiệt.
4. Giảm shared memory/thanh ghi để lên 3 block/SM.
5. Đo trên đồ thị chain thật (4575 node) thay vì Zephyr giả để hiệu chuẩn đúng độ sâu/quần thể.

## Quy tắc I/O (người dùng yêu cầu 11:07 14/11) — bảo vệ SSD
- Thư mục làm việc tạm RAM (tmpfs 20 GB): /home/vanthoi/lam-viec-tam. Build (target-opt), kết quả harness (CU4_OUTPUT=docs/perf/tmp/x.jsonl, symlink tới lam-viec-tam/perf-tmp; harness chấp nhận đường dẫn dưới lam-viec-tam), script và log trung gian: để ở lam-viec-tam/scratch.
- Chỉ những thứ quan trọng vào sao lưu (SSD): vài binary chọn lọc (hiện giữ opt5, opt8, opt17, opt19, opt22 và rc6), bản vá, ghi chú, TSV kết quả nhỏ (ceiling/).
- Đã xóa binary cũ trùng lặp trong lam-viec-tam-backup và /mnt/h/.../data; mỗi lần deploy chỉ copy đúng một binary vào /mnt/h/.../data (container cần), không copy thêm vào sao lưu trừ khi là bản tốt nhất.
- Đo đạc GPU: LUÔN dừng miner trước (người dùng nhắc lại 11:25 14/11: chạy song song tốn thời gian và sai lệch kết quả). Gom thành đợt ngắn, bật lại miner ngay sau khi xong. Việc không dùng GPU (đọc log, phân tích, biên dịch) làm khi miner đang chạy.

## Cách đo nhanh (harness)
`CU4_FAST=1 CU4_RUNS=1 CU4_CHUNK=32 CU4_SPB=32 CU4_BUDGET=custom CU4_READS=.. CU4_SWEEPS=.. CU4_BETA=lo,hi CU4_SEED=.. CU4_INST=.. CU4_SHAPE=12,4,62,1200 CU4_OUTPUT=docs/perf/x.jsonl` chạy test ignored `measure_generation_cost` (binary trong /home/vanthoi/lam-viec-tam/target-opt/release/deps/quip_solver_cuda-*). Dùng nhiều seed (3–6), lấy trung bình của min. Biến môi trường kernel/host: QC_PA_EVERY, QC_PA_FRAC, QC_SPB, QS_STAGES, QS_Q, QS_MIN, QS_WIN.
MỌI phép đo dùng GPU (kể cả chỉ đo năng lượng) phải dừng miner trước; không chạy song song.

## Bản screen của fork cũ (nonce-parallel, tầng 0 cấu trúc) — đánh giá 14/11 chiều
- Đo 300 instance (harness, năng lượng tầng B, 2 seed): 9 đặc trưng đếm loại tam giác; tương quan đơn ≤0,14; kiểm chéo 5-fold 0,276 (tiềm ẩn ≈0,36; trần nhiễu seed 0,77).
- Mô phỏng sim_s0.py (sự kiện/3 h ở mức -5079 ↔ mô hình -18,5): không S0 0,83; giữ 1/4: 2,03; 1/64: 3,1; 1/256: 4,13; 1/1024: 5,87. Ở -5085 (mô hình -26,9) ≈ 0 với mọi mức lọc.
- Kết luận: S0 chỉ tăng ×2–5 tần suất chạm ~-5079, KHÔNG đưa tới -5085 ⇒ theo tiêu chí mới KHÔNG triển khai/không viết prototype sample_lease cho đến khi có hướng đạt >-5085. API có sẵn (nếu cần sau này): Sampler::generates_locally/sample_lease, LeaseSink::push/screen/skip, FamilyLease::generate(nonce); CPU sinh ~1800 instance/s.

## Kiểm tra 13:46 VN 14/11 (opt26)
- Ta THẮNG 2 khối trong 60 phút: 12:53 (-5082, job 103 s) và 13:03 (-5084 log / -5083,07 raw, job 102 s = 2 dòng dõi); đối thủ 0xcd45a1: -5079, -5080, -5081 (3 khối). Cả 2 khối thắng của ta đến từ dòng dõi phụ (follow-up) ⇒ hướng follow-up CÓ HIỆU QUẢ, đạt -5083/-5084 (gần mục tiêu -5085).
- 7 job sâu/60 phút: tốt nhất -5083,07, trung vị -5074,0; 1 follow-up 562 s (-5075,6) bị hủy/không đạt. Miner restart 13:37 (do đo 300 instance) ⇒ cửa sổ sàng khởi tạo bằng prior.

## Quyết định tự chủ 13:50 VN 14/11 (người dùng: tự thực hiện, không hỏi)
- opt26: 12 job sâu/3 h; 2 job follow-up ngắn (103 s) đều thắng (-5081,3, -5083,1); 1 follow-up 562 s không đạt. Mẫu quá ít để chỉnh QS_PROMISE/QS_FOLLOW (nguy cơ làm nhiễu) ⇒ GIỮ opt26, thu thêm ≥3 h dữ liệu follow-up (thành công/hủy theo độ dài, năng lượng dòng dõi đầu vs cuối) rồi mới chỉnh. Không restart thêm.
- Việc kế (không cần GPU): thiết kế worm/loop có hướng trên siêu đồ thị tile/pair (hướng chưa thử #3), đo khi miner dừng theo đợt ngắn.

## opt27 (14:20 VN 14/11): ĐÃ BUILD, CHƯA TRIỂN KHAI — QS_PROMISE -5074 -> -5070
- Quan sát người dùng: job sâu lượt đầu chỉ -5071…-5075; 2 khối thắng (-5081,3; -5083,1) đều từ job 103 s (dòng dõi phụ); 0 khối thắng từ job sâu 1 dòng dõi cùng thời gian. Dòng dõi phụ rẻ (~52 s, không cần sàng) và là best-of-N trên instance đã qua sàng ⇒ nên chạy cho hầu hết job sâu (trước đây job -5071…-5073,9 không được follow-up).
- Giữ QS_EXIT=-5081 (job bị hủy mất kết quả), QS_FOLLOW=12. Binary: /home/vanthoi/lam-viec-tam/target-opt/release/quip-cuda-otsuka (sửa 1 dòng sampler.rs:380).
- Lệnh deploy bị trình phân loại quyền chặn (copy vào data + đổi symlink + restart); chờ người dùng cho phép. Miner vẫn chạy opt26.
- Xác nhận sau deploy: đếm follow-up, tỉ lệ ≤-5081, số bị hủy trong 3 giờ; không tăng thắng/giờ ⇒ về opt26.

- 14:25 VN 14/11: opt27 ĐÃ TRIỂN KHAI (người dùng cho phép). Symlink -> /data/quip-cuda-otsuka-opt27; rollback: ln -sfn /data/quip-cuda-otsuka-opt26. Theo dõi 3 giờ.

## opt27 THẤT BẠI — hoàn về opt26 lúc 14:41 VN 14/11
- Triệu chứng (dashboard): qblock 270 best -5044,3 (19 attempts), 271 best -5049,9 (127 attempts); 14:25–14:39 KHÔNG có job > 10 s nào; GPU 96% bận nhưng jobs_completed=0 nhiều phút liền.
- Nguyên nhân: QS_PROMISE=-5070 khiến gần như mọi job sâu chạy chuỗi dòng dõi phụ (tối đa 12 x ~52 s ≈ 10 phút) ⇒ GPU bị một job chiếm, sàng đói, round (4–10 phút) kết thúc → job bị hủy → mất toàn bộ kết quả; best mỗi round chỉ còn mức tầng sàng.
- Bài học: follow-up phải có giới hạn thời gian/số dòng dõi ngắn (≈2–3, ≤ ~2,5 phút) chứ không phải 12; chỉ mở rộng promise khi giới hạn độ dài đã chặt. 2 chiến thắng trước đều là job 2 dòng dõi (103 s). Đừng tăng QS_PROMISE mà không giảm QS_FOLLOW.

## Sự cố 15:05 VN 14/11 (opt26, QS_FOLLOW=12): job 637 s mất khối thắng + lỗi 'encode proof: family witness: length'
- Job ce60c4db (qblock 272, gen 3) chạy 637 s (follow-up tối đa 12 dòng dõi), best -5075,6, ngưỡng round -5066,8 ⇒ coordinator nhận (accepted) nhưng nộp thất bại 108 lần: 'submit error: encode proof: family witness: length' (codec từ chối số spin sai độ dài, ising-generators codec.rs). Chỉ xảy ra với job này trong cả lịch sử log; các job follow-up 103 s thắng bình thường. Chưa xác định nguyên nhân gốc (hàng spin n_valid=16 qua kiểm tra của validator nên nghi nonce/instance tái tạo không khớp với phần reads sau follow-up kéo dài); theo dõi.
- Cũng: job 637 s chiếm GPU ~10 phút, jobs_completed=0 trong round đó (round dài 15 phút vì đối thủ không thắng).
- opt28 (15:08): QS_FOLLOW 12 -> 2 (tối đa 3 dòng dõi ≈ 160 s), QS_PROMISE giữ -5074, EXIT -5081. Mục tiêu: tránh job siêu dài và rủi ro lỗi nộp.

## SỰ CỐ 15:00 VN 14/11 (opt26): khối thắng KHÔNG nộp được — "encode proof: family witness: length"
- Job ce60c4db… chạy 637 s (QS_FOLLOW=12 ⇒ ~12 dòng dõi), n_valid=16, best -5075,612, ngưỡng round đó -5066,76 ⇒ lẽ ra thắng. Coordinator báo `session submit failed (transient) … encode proof: family witness: length` rồi lặp 107 lần `win-time submit failed`. Chỉ 1 job từng gặp lỗi này (2 khối thắng trước bằng job 103 s nộp bình thường).
- Mã nguồn lỗi: ising-generators codec `check_spins`: hàng spin có độ dài != số nút của instance tái sinh từ nonce (hoặc không hợp lệ). Mỗi read đã được kiểm độ dài = graph.h.len() trong reads(), nên nghi: lệch giữa instance đã giải và nonce/instance coordinator tái sinh cho job rất dài (lease/nonce đổi trong lúc chạy 10 phút?). CHƯA tìm ra nguyên nhân gốc.
- Giảm thiểu: job dài là yếu tố chung ⇒ đặt QS_FOLLOW mặc định 3 (tối đa 4 dòng dõi ≈ 3,5 phút) — đã sửa mã + build (/home/vanthoi/lam-viec-tam/target-opt/release/quip-cuda-otsuka, "opt28"); lệnh deploy bị chặn quyền, chờ người dùng.
- Việc cần làm: tái hiện bằng cách ghi log độ dài hàng/nonce khi job > 5 phút; kiểm tra coordinator có đổi nonce của job khi lease quá hạn.
- Chính xác: mã nguồn hiện có QS_FOLLOW mặc định = 2 (tối đa 3 dòng dõi ≈ 2,6 phút), QS_PROMISE = -5074 (như opt26); binary opt28 build từ mã này, CHƯA deploy (bị chặn quyền). Miner vẫn chạy opt26 (QS_FOLLOW=12).

- 15:08 VN 14/11: opt28 ĐÃ TRIỂN KHAI (QS_FOLLOW=2, PROMISE -5074, EXIT -5081). Rollback: ln -sfn /data/quip-cuda-otsuka-opt26. Theo dõi: khoảng trống attempt tối đa (~3 phút), lỗi 'family witness: length' còn không.

## PHÁT HIỆN 15:15 VN 14/11: ngưỡng RÃ (gated decay) trong round + vòng nhanh ~10 s của đối thủ
- Ngưỡng round (threshold_milli trong attempts.jsonl) rã dần khi round không có người thắng: qblock 272 bắt đầu -5079,7, sau 13 phút còn -5066,8 (lúc job 637 s xong). Sau khối thắng, các round kế tiếp bắt đầu thấp (-5065,3 ở qblock 273) rồi leo dần theo mỗi khối thắng (-5066,3; -5067,3; -5068,1; -5068,2; -5070,2; -5071,0 ở 274–279), mỗi round chỉ 8–25 s.
- Đối thủ 0xcd45a1 thắng 4 round liên tiếp lúc 15:02–15:03 bằng attempt ~10 s với nonce -5067…-5072 (đúng ở ngưỡng). Ta trong các round đó: best -5061,2 / -5056,4 / -5061,9 / -5062,0 (qblock 273–276), thiếu 4–6 đơn vị; job sâu 54 s luôn bị hủy khi round kết thúc sau 10 s.
- Hệ quả: (1) khi ngưỡng thấp, cần một lượt NHANH (~10 s) đạt -5067…-5071: chất lượng ở ngân sách 10 s quyết định; tầng C512x16384 (3,5 s, trung vị -5059,6) và deep (54 s, trung vị -5073) có khoảng trống ở giữa. Ý tưởng chưa thử: tầng D 512 reads x 32768–65536 sweeps (~7–14 s) cho instance qua C; hoặc rút ngắn deep khi round đang rẽ nhánh nhanh. (2) Job siêu dài (637 s) vừa mất cơ hội thắng round đó vừa để lỡ chuỗi round nhanh.
- Cần đo khi miner dừng: năng lượng vs thời gian (5–20 s) ở instance đã qua sàng C để xác định tầng D.

## Kiểm tra 15:33 VN 14/11 (opt28 đang chạy từ ~15:23, QS_FOLLOW=2)
- THAY ĐỔI CHAIN: ~15:07 VN topology đổi 4575/41474 → 0x2d130c… 4560 nodes/41205 edges; 15:04–15:11 đối thủ 0xcd45a1 thắng ~13 khối liên tiếp (ngưỡng leo -5071→-5081, vòng chỉ vài chục giây); từ 15:11 ngưỡng ≈ -5080/-5081. Phải hiệu chuẩn lại prior (QS_PRIOR) cho đồ thị 4560 nút nếu năng lượng đổi.
- opt28 job sâu: 15:12 53 s -5072,2; 15:25 151 s -5076,6; 15:28 149 s -5078,6; 15:32 149 s -5080,05 (đạt ngưỡng -5080 nhưng bị pallet từ chối: 'no qblock was recorded at this block' đúng lúc vòng 293 bắt đầu 08:32:00Z, cancelled_jobs=2 — nghi lệch vòng/đua nộp, không phải lỗi witness).
- Lỗi 'family witness: length' 15:00 (job 637 s) vẫn chưa rõ nguyên nhân; chưa lặp lại với job ≤151 s.

## ĐO 15:20 VN 14/11 (miner dừng 15:15–15:20 và 15:21–15:22): năng lượng vs thời gian + tầng sàng siêu rẻ (ceiling/tierD.tsv, cheap.tsv)
- Harness, instance 0/3, seed 11 (trung bình 2 instance): 512x16384 3,4 s -5671,9; 2048x16384 6,1 s -5675,5; 2048x32768 12 s -5684,1; 2048x65536 24 s -5687,9; 2048x131072 48 s -5689,0. ⇒ deep 65536 (24 s) chỉ kém 131072 ~1,1 harness (~0,8 live) nhưng nhanh gấp đôi; 32768 (12 s) kém ~5 harness (~3,5 live) — KHÔNG có tầng 'D' đáng giá giữa C và deep cho vòng 10 s.
- Tầng sàng rẻ (20 instance, 1 seed, tương quan với C512 trung bình 2 seed): 16 sweep x64 reads 0,006 s ρ 0,06; 64x64 0,011 s ρ 0,43; 256x64 0,03 s ρ 0,52; 64x256 reads 0,023 s ρ 0,52. So với A256 hiện tại (1024 sweep x256 reads, 0,13–0,19 s) ρ(C512) = 0,50 ⇒ cùng khả năng xếp hạng nhưng RẺ HƠN 5–8 lần (n=20, sai số ±0,2; cần xác nhận).
- Phân tích tỉ lệ thành công: đối thủ ~5%/attempt 10 s ở -5080 (thời gian thành công TB ~3 phút), ta ~8%/job sâu nhưng ~1 job sâu mỗi ~4 phút (80% GPU ở sàng) ⇒ chênh ~15x về số lần thử chất lượng/phút. Đòn bẩy chính = hiệu quả sàng (số instance/giây GPU), tức bản screen nonce-song-song của fork cũ (14–17 triệu nonce/s trên chain cũ) là hướng đúng nhưng cần chuyển sang family mới.

## Kiểm tra 15:35 VN 14/11 (opt28, chạy từ 15:08)
- THẮNG 15:33:58 với nonce -5084 (job sâu 53 s, 1 dòng dõi; log -5084, raw -5083,9). Job 15:32:06 (follow-up 149 s, -5080,0 ≤ ngưỡng, accepted) nộp nhưng pallet từ chối: 'no qblock was recorded at this block' ⇒ mất cuộc đua submit (lần đầu thấy kiểu trùng/muộn khối; chủ đề 'nộp sớm/thứ tự' đã hoãn đến khi có khối trùng — nay có ví dụ, cần điều tra: dòng log 08:32:06Z, qblock 293, job 544a993a).
- 7 job sâu/60 phút: tốt nhất -5083,9; trung vị -5076,6; 3 follow-up 149–151 s (-5076,6; -5078,6; -5080,0). Khoảng trống attempt tối đa 284 s (so với 682 s ở opt26). Không còn job siêu dài, không còn lỗi 'family witness: length' mới.

## opt29 (15:52 VN 14/11): tầng A siêu rẻ — QS_STAGES 256:64:64;256:8192:16;512:16384:8, QS_PRIOR -4800;-5057;-5068
- Mô phỏng (ceiling/sim_s2.py, ngân sách 600 s/round, deep 48 s, ρ(S64x256)=0,39 với deep, 0,023 s): S64/q64→B256/q16→C512/q8: -16,14 (harness, tốt nhất mỗi round) so với baseline A256 q8/B256 q8/C512 q4: -14,11 ⇒ +2,0 harness ≈ +1,4 live. Khiêm tốn, KHÔNG đủ để chạm -5085 (cần thêm hướng lớn), nhưng rẻ để thử.
- Rủi ro (bài học opt22–24): chi phí live của A có thể cao hơn harness (overhead job/lease/chuẩn bị đồ thị). Kiểm tra sau 15 phút: job/s, số job sâu mỗi 10 phút (opt28 ~1 mỗi 3–4 phút), best mỗi round. Nếu deep/10 phút giảm hoặc job/s bất thường ⇒ hoàn về opt28: ln -sfn /data/quip-cuda-otsuka-opt28.

- 15:50 VN 14/11: opt28 vừa thắng 3 khối liên tiếp (15:33 -5084, 15:42 -5081, 15:43 -5083); tôi triển khai opt29 lúc 15:47 mà không kiểm tra log ⇒ người dùng nhắc. HOÀN về opt28 15:50. opt29 (tầng A rẻ, +1,4 live mô phỏng) hoãn đến khi opt28 không còn thắng; quy tắc: kiểm tra log trước khi deploy.

- 16:02 VN 14/11: kiểm tra log trước deploy — opt28 không thắng từ 15:43 (18 phút), miner không treo (attempt mới 20 s trước; khoảng trống ~160 s = follow-up 3 dòng dõi). DEPLOY opt29 (tầng A rẻ). Kiểm tra sau 15 phút: job/s, số job sâu/10 phút, best mỗi round; hoàn về opt28 nếu kém.

## opt29 KẾT QUẢ XẤU → opt30 (16:12 VN 14/11)
- opt29 (A 64 sweep x256 reads q64, B q16, C q8): 7,4 phút đầu: 10946 attempts (24,8/s, ~1,2 MB/phút ≈ 1,7 GB/ngày ghi attempts.jsonl so với ~58 MB/ngày ở opt28), quip-cuda CPU 262%; B 169 lần, C 5 lần, deep 0 (opt28 ~1 mỗi 3–4 phút).
- LỖI MÔ HÌNH: sim_s2.py/simulate_pipeline.py dùng phân vị TUYỆT ĐỐI của N(0,1) cho mỗi tầng, trong khi live xếp hạng trong cửa sổ của CHÍNH tầng (chỉ gồm job đã vào tầng đó) ⇒ tỉ lệ qua thực = đúng 1/q, không cao hơn do tương quan. Do đó chi phí mỗi deep thật = Π q x chi phí: opt28 (A 0,15 s, q8/8/4): 256A+32B+4C ≈ 81 s; opt29 (q64/16/8): 8192A+128B+8C ≈ 307 s (4 lần đắt). Mô phỏng +2,0 là SAI; lặp lại lỗi opt22–24 (đừng dùng sim này cho cấu hình q). Cần sửa sim: quantile theo cửa sổ có điều kiện.
- opt30: A 64 sweep x256 reads q16, B 8192 q8, C 16384 q4, prior -4795;-5052,5;-5064. Chi phí mỗi deep ≈ 512A(0,023 s)+32B+4C ≈ 53 s (opt28: 81 s) với độ chọn lọc 512 (opt28: 256) ⇒ kỳ vọng nhiều deep hơn ~1,5x và chọn lọc gấp đôi. Tác dụng phụ: ~25 attempts/s ghi dashboard (ghi đĩa, tải coordinator) — cần giảm bằng cách chuyển sang sampler cấp lease (sink.screen thay vì attempt từng salt) nếu đo cho thấy quá tải.
- Kiểm tra sau 15 phút: deep/10 phút ≥ opt28, best mỗi round, CPU, tốc độ ghi attempts.

## opt30 (chạy từ 16:10 VN 14/11) — 15 phút đầu: THẮNG -5083 lúc 16:19:47
- 2835 attempts (3,0/s; opt29 25/s), 4 job sâu: 148 s -5078,3; 100 s -5082,6 (THẮNG, nonce -5083); 148 s -5079,2; 52 s -5073,3. Best mỗi round: 299 -5057,6; 300 -5082,6; 301 -5079,2. CPU 137%. Khoảng trống attempt tối đa 160 s.
- So với opt28: nhịp job sâu tương đương (1 mỗi ~3–4 phút) nhưng chất lượng tốt hơn (3/4 job -5078…-5083). Mẫu nhỏ; GIỮ, không đổi, theo dõi vài giờ rồi mới thử: deep 65536 sweep, chỉnh QS_PROMISE.
- Lưu ý đọc dashboard: dòng 'Best Energy/Attempts' theo qblock gộp cả hai bản nếu deploy giữa round ⇒ luôn tách theo timestamp.

## ĐIỀU TRA SỰ CỐ NỘP (16:40 VN 14/11) — kết luận
1. 'family witness: length' (job ce60c4db, qblock 272): NGUYÊN NHÂN = ĐỔI TOPOLOGY GIỮA CHỪNG. Log 'new round' cho thấy topology 0x909cfee7… (4575 nút/41474 cạnh) dùng từ qblock 105 đến 272, rồi từ qblock 273 (08:02:52Z) là 0x2d130ce6… (4560 nút/41205 cạnh). Job chạy trên đồ thị 4575 nút (spin 4575) nhưng coordinator mã hóa với topology mới 4560 ⇒ codec từ chối 'length'. Không liên quan độ dài follow-up hay mã của ta. Round 272 kéo dài 15 phút (không ai thắng, ngưỡng rã) cũng khớp với việc chuỗi đổi topology trong round. Hệ quả: job bắt đầu trước lúc đổi topology và nộp sau đó sẽ lỗi — không tránh được; coordinator tự dừng thử lại khi sang round mới.
2. Pallet từ chối 15:32:06 (job 544a993a): job follow-up 149 s, thế hệ 1 của qblock 293, ngưỡng -5079,347, kết quả -5080,045 (ĐẠT) nhưng về đích lúc 08:32:06Z, 6 s SAU khi round sang thế hệ 2 (08:32:00Z) ⇒ 'proof included but ExtrinsicFailed: no qblock was recorded at this block'. Là cuộc đua ranh giới round, không phải lỗi: thua sát nút vì follow-up không trả sớm. Cải tiến khả dĩ (chưa làm, bản đang thắng nên không đổi): QS_EXIT động theo ngưỡng round (ngưỡng thực -5079,3…-5081,6; QS_EXIT=-5081 cố định khiến lúc ngưỡng dễ vẫn chạy tiếp thêm dòng dõi thay vì nộp ngay). Miner (đường stream) KHÔNG nhận ngưỡng; chỉ đường lease có LeaseSink::target_energy_milli ⇒ cần sampler cấp lease để biết ngưỡng.
3. Topology mới 4560/41205: các hiệu chỉnh (harness 4800 nút, prior năng lượng) vẫn dùng được; nonce thắng vẫn -5081…-5084.

## opt32 (17:10 VN 14/11): SAMPLER CẤP LEASE + SÀNG CẤU TRÚC CPU (hướng screen của fork cũ, nay cho chain 120)
- Mã: src/cluster/sampler/lease_run.rs (+ bản vá opt32-lease-sampler.patch, lease_run.rs trong sao lưu). `generates_locally()=true`, `sample_lease`: QS_GEN=24 luồng CPU sinh salt (FamilyLease::generate), điểm cấu trúc (mô hình tuyến tính 9 đặc trưng mật độ tam giác/cạnh, ceiling/s0_model.json; Rust khớp Python <1e-6) xếp hạng trong cửa sổ 2048, qua 1/q (q thích nghi từ QS_S0Q=16: GPU đói → nới, GPU luôn có việc → siết); một luồng GPU chạy pipeline A/B/C/deep hiện có (một lease mỗi lần qua LEASE_GATE vì coordinator phát hàng chục lease 4500–17000 salt cùng lúc). Salt bị loại: LeaseSink::screen (không tạo kết quả/attempt/ghi dashboard). Chỉ job sâu mới push. Ngưỡng thoát follow-up = target round động (LeaseSink::target_energy_milli) trừ QS_MARGIN=0,3.
- Kết quả ban đầu: ~200–295 salt/s (opt30: ~25/s), GPU 99%/445 W, CPU ~180–1950%, 0 lỗi. Chưa có job sâu trong 3 phút đầu; đánh giá sau 15–30 phút: job sâu/10 phút (opt30: 2,25), trung vị, best mỗi round. Hoàn về opt30 nếu kém: ln -sfn /data/quip-cuda-otsuka-opt30.
- opt31 (chạy 1 phút) lỗi thiết kế: coordinator phát >100 lease song song, mỗi lease một luồng GPU + 10 luồng sinh ⇒ ~1000 luồng; opt32 thêm LEASE_GATE.

## opt32 KẾT QUẢ 22 PHÚT (17:10–17:33 VN 14/11): THẮNG 2 KHỐI (17:21 -5082; 17:32)
- 7 attempt (job sâu push; device_access_time=0 trong đường lease nên không lọc theo thời lượng): -5069,8; -5078,4; -5074,3; -5081,8 (thắng); -5076,4; -5079,5; -5079,6 ⇒ trung vị -5078,4, ~3,5 job sâu/10 phút (opt30 2,25). Best mỗi round 306 -5078,4; 307 -5081,8; 308 -5079,6. GPU 99%/441 W, CPU ~186%, 0 lỗi. ~200–295 salt/s.
- GIỮ opt32. Chưa đạt -5085 ổn định ⇒ chưa dừng tối ưu. Tiếp theo (sau vài giờ dữ liệu): QS_S0Q, QS_MARGIN, QS_GEN, thêm đặc trưng cấu trúc.

## Dashboard: dòng khối thắng bị trùng (14/11 17:55 VN) — ĐÃ SỬA TRONG MÃ NGUỒN COORDINATOR, CHƯA TRIỂN KHAI (chờ bản miner/coordinator mới theo yêu cầu người dùng)
- Triệu chứng: qblock 303 và 309 hiện 2 dòng: 'GPU-CUDA rejected (N attempts)' và 'GPU submitted_inblock (chain-only, attempts —)'.
- Nguyên nhân: nonce thắng là kết quả bị stash (không đạt ngưỡng lúc job xong, ghi attempt accepted=false/submitted=false), rồi ngưỡng rã/đổi round và đường 'win-time submit' (runtime.rs) nộp nó thành công — nhánh Success chỉ mark_submitted + cập nhật summary, KHÔNG ghi attempt submitted=true kèm extrinsic/khối ⇒ dashboard không ghép được với sự kiện chain, nên hiện thêm dòng chain-only. Các khối thắng nộp ngay trong session (vd #308) có attempt submitted=true nên ghép đúng.
- Sửa: crates/quip-coordinator/src/attempt.rs thêm AttemptRecord::win_time(...); runtime.rs nhánh Success của win-time ghi WriterMsg::Attempt (accepted=true, submitted=true, extrinsic/block/pow_sequence từ SubmitReceipt) cùng Summary. Biên dịch `cargo check` qua. Bản vá: coordinator-wintime-attempt.patch (sao lưu). Binary đang chạy là coordinator 0.3.8-rc10 của upstream (không build từ cây này, cây là rc9) ⇒ áp dụng khi có phiên bản mới: patch vào cây mới, build, thay /usr/local/bin/quip-coordinator trong container. Dòng trùng cũ trên dashboard không tự mất.

## opt32 đang thắng liên tiếp (17:56 VN 14/11) + vì sao dashboard thiếu dòng #311
- opt32 thắng #309 (17:32), #310 (17:42), #312 (17:51, -5081), #313 (17:54, -5082): 4 khối trong ~25 phút. KHÔNG deploy gì khác khi đang thắng.
- Thiếu dòng #311 trong Recent Performance: không phải lỗi từ chối. Thư mục attempts/<N> ứng với dòng #N+1; round tạo #311 (17:43–17:48) KHÔNG có bản ghi attempt nào (không có thư mục attempts/310) vì opt32 chỉ push job sâu, còn salt bị loại dùng LeaseSink::screen (không tạo kết quả/attempt) — round đó không có job sâu hoàn tất (lease done chỉ có best tầng sàng -5053…-5067). Dashboard chỉ liệt kê qblock có ít nhất một attempt. Vô hại về chức năng (round đó đối thủ #311 tu17nodeslove thắng lúc 17:48).
- opt33 (đã build, CHƯA deploy; /home/vanthoi/lam-viec-tam/scratch/quip-cuda-otsuka-opt33): ngoài job sâu, push thêm kết quả sàng khi nó cải thiện best của lease (~10 bản ghi/lease) ⇒ mỗi round luôn có dòng trên dashboard, và kết quả sàng đã đạt ngưỡng cũng được nộp. Deploy khi opt32 hết thắng liên tiếp (quy tắc kiểm tra log).

## Quyết định ghi attempt (18:08 VN 14/11, người dùng): chỉ ghi các nonce tốt nhất, không ghi nhiều
- Giữ chính sách của opt32: chỉ job sâu push (1–3 bản ghi/round), salt bị loại dùng screen. opt33 (push thêm ~10 bản ghi/lease khi cải thiện best) BỊ HOÃN/BỎ — không deploy. Chấp nhận round không có job sâu sẽ không có dòng trên dashboard (vd #311).

## HUẤN LUYỆN BỘ DỰ BÁO CẤU TRÚC (18:40 VN 14/11) — KẾT QUẢ ÂM: không cải thiện được ρ
- Dữ liệu: 300 instance harness (scan/), nhãn = trung bình năng lượng tầng B (256x8192) của 2 seed (nhiễu: tương quan seed-seed 0,428 ⇒ trần lý thuyết ρ ≈ 0,77 với trung bình 2 seed).
- Đặc trưng v1 (9 mật độ tam giác phân loại + cạnh yếu dương): CV 5-fold ρ 0,238 (ridge). Đặc trưng v2 (chu trình 4 bị frustration toàn bộ/chỉ cạnh yếu, bậc cạnh yếu, thống kê ghép tile–tile, tỉ lệ nút trong cặp): ρ −0,003 riêng lẻ; v1+v2 ridge 0,196, GBR 0,207, RF 0,184 — KHÔNG hơn v1. Đặc trưng đơn tốt nhất v2: fr4 0,14, dw_sd −0,13.
- Kết luận: bộ dự báo đếm cấu trúc bão hòa ở ρ ≈ 0,24 (≈ 0,31 so với biến tiềm ẩn); đừng thử thêm đặc trưng đếm tương tự. Muốn cao hơn cần dự báo dựa trên động lực học (anneal thật), mà tầng A GPU (64x256, ρ 0,52, 0,023 s) đã làm tốt hơn. Mã: ceiling/feat2.py, fit2.py; môi trường scratch/venv (numpy/scipy/sklearn).
- Hệ quả cho -5085: ước tính +1…+1,5 từ bộ dự báo tốt hơn KHÔNG có cơ sở; đòn bẩy còn lại ≤ ~+0,6 (cân lại q ~+0,3; kernel 3 block/SM ~+0,3). Nhịp -5085 hiện ~1 lần mỗi ~2 giờ (xác suất ~3%/job sâu).

## ĐÍNH CHÍNH (18:36 VN 14/11, người dùng): log làm tròn xuống — nonce thật sâu nhất opt32 = -5084,115, CHƯA đạt -5085
- Dòng 'stopping miners ... best_energy' hiển thị floor của năng lượng (-5084,115 → "-5085"). Giá trị thật (attempts.jsonl): #313 -5082,225; #314 -5083,205; #315 -5084,115; #317 -5082,585. Mọi con số 'nonce' lấy từ log trong các báo cáo trước lệch tối đa 1 đơn vị (nông hơn thực tế đã công bố ~0–1… tức thực tế nông hơn log).
- Ngưỡng round kế sau khối thắng = đúng năng lượng nonce thắng (round 316 thr -5084,115), rồi rã dần.
- Mục tiêu -5085: CHƯA đạt; vẫn tiếp tục tối ưu. Luôn dùng attempts.jsonl để báo cáo.

## CHỈ THỊ NGƯỜI DÙNG (18:40 VN 14/11): TẠM DỪNG TỐI ƯU, CHỈ GIÁM SÁT
- Tối ưu lại CHỈ khi ta không còn thắng quá 5 khối (hiểu: ≤ 5 khối thắng trong cửa sổ 60 phút; opt32 đang 6–8/giờ). Trong lúc đó: không deploy, không thử nghiệm GPU; chỉ đọc log/giám sát, và các việc không đụng GPU (mã coordinator chờ deploy).
- Đã sửa log coordinator (logging.rs: energy_units giờ in 3 chữ số thập phân thay vì làm tròn xuống) — nằm chung trong coordinator-wintime-attempt.patch (attempt.rs win_time + logging.rs + runtime.rs); 334 test lib qua; bản release v0.3.9-rc5 đã vá build sẵn ở target-coord. CHƯA triển khai (chờ cập nhật miner/coordinator mới theo yêu cầu trước đó).

## ĐÍNH CHÍNH CHỈ THỊ (18:45 VN 14/11): tối ưu lại CHỈ khi 5 KHỐI LIÊN TIẾP không có khối nào do ta thắng
- (Thay cho cách hiểu '≤5 khối/giờ'.) Đếm các qblock liên tiếp gần nhất do đối thủ thắng kể từ khối thắng của ta (signer ta 0xd2ec60); ≥5 ⇒ được tối ưu/deploy (sau khi kiểm tra log); <5 ⇒ chỉ giám sát.

## ĐỘNG HỌC NGƯỠNG (phân tích CPU 19:15 VN 14/11, từ threshold_milli trong attempts.jsonl 16:00–19:14)
- Sau mỗi khối thắng: ngưỡng round kế = max(ngưỡng round trước − 1,000; năng lượng nonce thắng). Ví dụ: 313 thr -5081,145 → 314 -5082,145 → 315 -5083,145 → ta thắng -5084,115 ⇒ 316 thr -5084,115 (bị chặn bởi nonce thắng); 320 thr -5080,675, ta thắng -5084,615 ⇒ 321 thr -5081,675 (chỉ −1).
- Trong round, ngưỡng RÃ dần theo thời gian (gated decay) cho tới khi có người đạt (round 316: -5084,115 lúc 18:11 → đối thủ thắng ~-5081,9 lúc 18:23).
- Hệ quả: (1) chuỗi thắng liên tiếp khiến ngưỡng siết 1 đơn vị mỗi khối (thang -5080,8 → -5084,1 trong 4 khối liên tiếp 17:51–18:07); chuỗi dừng khi ta không đạt bậc kế. Nonce sâu hơn mức cần KHÔNG làm round sau khó hơn quá 1 đơn vị ⇒ không có đòn bẩy chiến lược kiểu 'nộp nông hơn'. (2) Round khó (sau chuỗi thắng) được quyết bởi đuôi phân bố job sâu (ai có nonce ≤ ngưỡng đang rã trước) ⇒ đòn bẩy vẫn là đuôi sâu/thông lượng job sâu.

## KHẢO SÁT HƯỚNG MỚI (19:40 VN 14/11, chỉ CPU): đọc kernel + quip-solver-cpu, thử SA đơn spin rẻ làm tầng sàng
- Đọc kernels/cluster.cu: thuật toán = cụm FK trên lớp mạnh (liên kết tile 2-up/2-down + cặp dimer, xác suất bond theo beta·J) + heat-bath quay vòng trên cụm theo trường yếu (rounds) + Metropolis đơn spin mỗi sweep; PA ngoài kernel. Cấu trúc đồ thị (file scan: mục T = 2116 tile 4 nút CHỒNG LẤP — mỗi nút thuộc ≤2 tile ⇒ spin nằm trên CẠNH, tile là ĐỈNH: mô hình ice/6-đỉnh, quy tắc 2-vào-2-ra; mục P = 1200 cặp).
- Đọc quip-solver-cpu: có kernel chuyên họ ice/dimer (dimer_weighted/fractional, tileworm, pairworm, ice_sat = Kissat giải lớp mạnh, flow/flow_loops, routing, gadget_repair) và các họ SB (bsb/dsb/hbsb/hdsb/gbsb/tedsb/sbqa), Flatiron/MPS (mạng tensor). Chưa thử trên GPU: SB các biến thể, move vòng có hướng chuẩn (directed loop), Houdayer ICM (nghi không hiệu quả: bậc 20 ⇒ cụm chồng lấp thấm sớm).
- THỬ (CPU, 300 instance, tương quan với năng lượng tầng B): SA đơn spin chromatic rẻ (16 replica; beta 0,3→3): 16 sweep ρ 0,19; 32 sweep ρ 0,30 (0,04 s/inst trên 6 tiến trình); 64 sweep 0,22; 128 sweep 0,28; beta cao hơn kém (0,05–0,16). KẾT QUẢ ÂM/BẰNG: ρ ≈ 0,2–0,3 ngang bộ dự báo cấu trúc (0,24); không phải tầng sàng đột phá. (Lần chạy 40 instance cho 0,39 là nhiễu mẫu nhỏ.)
- Chưa có hướng phần mềm bứt phá nào được chứng minh. Mô phỏng cân q (sim_lease.py, CPU) vẫn chạy.

## MÔ PHỎNG sim_lease.py (CPU, mô hình cửa sổ đúng, 3 lần x 1 giờ mô phỏng mỗi cấu hình) — XONG 20:42 VN
- Hiệu chuẩn: mu -5068,55; cấu hình live (qS0 16, qA 16, qB 8, qC 4, follow 2) mô phỏng ra 13,3 job sâu/giờ, trung vị -5080,0 (thật: ~16/giờ, -5079,7 ⇒ khớp tốt).
- Ứng viên tốt nhất theo P(≤-5085)/giờ: (qS0 64, qA 16, qB 4, qC 8, follow 1): P85 1,0/giờ, P83 2,67/giờ, 16,7 job sâu/giờ, trung vị -5079,9, cần nguồn 219 instance/s (≤ ~250/s CPU cung cấp được). Live: P85 0,33/giờ, P83 0,67/giờ.
- CẢNH BÁO: mẫu quá nhỏ (đếm 1–3 sự kiện/3 giờ) ⇒ nhiễu lớn; sim_confirm.py chạy lại các ứng viên với 40 giờ mô phỏng mỗi cấu hình để xác nhận trước khi đổi q.

## opt34 TRIỂN KHAI 20:48 VN 14/11 (người dùng yêu cầu thử thực tế; quyết định sau 2 round: giữ hay về opt32)
- Thay đổi: PreparedGraph::from_instance + ising_graph được dựng trong các luồng sinh (CPU rảnh) thay vì trên luồng GPU; screened_reads nhận prepared sẵn cho tầng đầu. Lý do: GPU chỉ 63% (rảnh ~37%, từng đợt 0,2–4 s) vì luồng GPU chuẩn bị đồ thị tuần tự (~9 ms/nonce) trước kernel A 23 ms. Kỳ vọng thông lượng +~20%.
- Chỉ số quyết định sau 2 round (~20 phút): mức dùng GPU TB (opt32: 63–65%), salt/s mỗi lease, số/chất lượng attempt (opt32: ~16 job sâu/giờ, trung vị -5079,7), thắng. Về opt32: ln -sfn /data/quip-cuda-otsuka-opt32 rồi docker restart quip-cuda.

## opt35 (20:52 VN 14/11): SỬA LỖI CỦA opt34 — nó vô tình mang logic push-khi-cải-thiện của opt33
- Người dùng thấy ~12 attempt/round với opt34. Nguyên nhân: tôi đã BỎ opt33 (người dùng chỉ muốn ghi nonce tốt nhất) nhưng KHÔNG gỡ đoạn 'push kết quả sàng khi cải thiện best của lease' khỏi lease_run.rs, nên opt34 build từ file đó vẫn ghi ~10 bản ghi/lease (cả kết quả sàng -5049…-5063 và salt đầu lease -4841). Đã gỡ: opt35 = opt34 (chuẩn bị đồ thị song song) + CHỈ push job sâu. 
- Bài học: khi bỏ một thay đổi chưa deploy, phải hoàn tác mã nguồn ngay, không để lẫn vào bản build kế.
- Đánh giá 2 round (từ 20:52): attempt chỉ là job sâu (1–3/round), GPU dùng cao hơn opt32 (63%), không lỗi.

## opt36 TRIỂN KHAI 21:22 VN 14/11 (người dùng duyệt, opt35 vừa thắng 2 khối): QUẦN THỂ DEEP 4096 + FOLLOW-UP 1
- Cơ sở: ceiling/stage2.tsv (65536 sweep, instance 0/3, seed 11/22): quần thể 4096→256 @0,35 đạt −5690…−5692 vs 2048→256 −5685…−5686.5 (≈ −5 harness ≈ −3,5 live) với ~2x thời gian (101 vs 54 s). Mẫu nhỏ; các ghi chú cũ nói 2048 ≈ hòa vốn ⇒ cần đo live.
- Thay đổi: CUDA_ADAPT min/max reads 2048→4096 (src/lib.rs); QS_FOLLOW mặc định 2→1 (mỗi dòng dõi ~100 s). Bản vá: opt36-pop4096.patch. opt35 là bản nền (GPU 99%, đang thắng).
- Quyết định sau 2 round (~21:45): trung vị năng lượng job sâu so với opt35/opt32 (−5079,7); giữ nếu trung vị tốt hơn ≥ ~1,5 (job sâu sẽ ít hơn ~2x); không thì quay về opt35: ln -sfn /data/quip-cuda-otsuka-opt35 rồi docker restart quip-cuda.

## COORDINATOR ĐÃ TRIỂN KHAI (21:22 VN 14/11, người dùng yêu cầu): v0.3.9-rc5 đã vá
- Binary: /data/quip-coordinator-0.3.9-rc5-patched (build từ cây v0.3.9-rc5 + coordinator-wintime-attempt.patch: attempt win_time + log năng lượng 3 chữ số thập phân). Bản cũ lưu: /data/quip-coordinator-0.3.8-rc10.orig. Symlink trong container: /usr/local/bin/quip-coordinator -> /data/quip-coordinator-0.3.9-rc5-patched.
- Khởi động OK: protocol 2, validator RPC (spec_version 120, quantum_pow_api 3) kết nối, tải topology 4560 nút, miner cuda-0 đăng ký, dashboard :8086.
- Quay lại: docker exec quip-cuda ln -sfn /data/quip-coordinator-0.3.8-rc10.orig /usr/local/bin/quip-coordinator rồi docker restart quip-cuda.
- Lưu ý: đồng thời đang thử opt36 (21:22) — kết quả opt36 sau 21:25 chịu thêm một lần khởi động lại; coi 21:23 là mốc bắt đầu.

## ĐÁNH GIÁ opt35 (21:25 VN 14/11, 20:52:30–21:21:45, 29 phút) + NONCE ≤ -5085 ĐẦU TIÊN
- MỤC TIÊU -5085 ĐÃ ĐẠT MỘT LẦN: opt32 thắng #333 lúc 20:38:30 với nonce THẬT -5085,465 (ngưỡng -5080,475); #334 lúc 20:41:40: -5083,125. (opt32 12 khối thắng/218 phút.)
- opt35 vs opt32: GPU 99% vs 63–65% (441 W vs 288 W); job sâu 3,1 vs 2,1 mỗi 10 phút (+48% ≈ dự đoán +45%); trung vị -5077,9 vs -5079,8 (9 mẫu, sai số ±1,1, trong 20 phút sau 2 lần restart ⇒ chưa kết luận); tốt nhất -5083,755; 2 khối thắng (21:10 -5082,855; 21:17 -5083,755) ≈ 4,1/giờ vs 3,3/giờ. 0 lỗi. KẾT LUẬN: giữ opt35 làm bản nền (cải thiện thông lượng thật).

## KẾT QUẢ sim_confirm.py (22:10 VN 14/11; mô hình cửa sổ đúng, 40 giờ mô phỏng/cấu hình, mu -5068,55)
- Live (qS0 16, qA 16, qB 8, qC 4, follow 2): 14,7 job sâu/giờ; P(≤-5083) 1,00/giờ; P(≤-5085) 0,175/giờ; trung vị -5080,0; nguồn 56/s.
- Khả thi (nguồn ≤ ~250/s): (64,16,4,8,1) P83 1,32 P85 0,200; (64,16,8,2,1) P83 1,20 P85 0,200 (21,9 job sâu/giờ). Không khả thi (nguồn quá lớn): (64,32,16,4,2) 663/s P85 0,375; (64,32,8,2,1) 357/s P85 0,300.
- KẾT LUẬN: đổi q chỉ cho +14%…+32% ở mức ≤-5083/-5085 (nhiễu ±~35%); KHÔNG đáng deploy. Hướng cân q coi như đã khai thác xong.
- opt36 47 phút: 12 attempt, tốt nhất -5084,385, trung vị -5079,23, 2 khối thắng (21:40 -5084,385; 22:01 -5082,565); chưa đạt tiêu chí (+1,5). Đánh giá lại ~22:45.

## RÀ SOÁT MÃ SÂU (22:25 VN 14/11, theo yêu cầu 'đọc kỹ xem còn phần nào bỏ qua') — phát hiện chưa khai thác
1. KERNEL BỊ GIỚI HẠN ĐỘ TRỄ, KHÔNG PHẢI BĂNG THÔNG: số đo cũ cho thấy gấp đôi khối chỉ tốn +14–30% thời gian (A64 0,10 s vs A256 0,13 s; B128 0,77 s vs B256 0,90 s; C256 1,77 s vs C512 3,45 s = 2 đợt). Mỗi SM chỉ chạy 2 khối vì shared 9 B/nút (parent int32 + acc int32 + s int8 = ~41 KB/khối; Ada 100 KB/SM). Ghi chú cũ ước '3 block/SM chỉ +10–15%' dựa trên occupancy thread; nhưng threads/block 128–1024 'không đổi' = hiệu năng phụ thuộc số KHỐI độc lập hơn số thread. Ước lượng mới: parent int16 + 256 thread/khối (7 B/nút ≈ 32 KB, 3 khối/SM) ⇒ +25–35% thông lượng (~+0,3 điểm). Cần: sửa cluster.cu (parent/find/unite short), CLUSTER_SHARED_BYTES_PER_NODE (capacity.rs) và đo trên GPU trống.
2. VÒNG PA ĐỒNG BỘ VỚI HOST MỖI 32 SWEEP (device.rs 424–538): mỗi bước = synchronize + alloc_zeros + kernel năng lượng + dtoh + sort CPU + htod map + kernel resample. Chạy cả ở tầng B (256 bước/lần), C (512), deep (4096). Ước ~0,3–0,5 ms GPU rảnh/bước ⇒ ~14% tầng B, ~7% tầng C, ~3–6% deep ⇒ ~7% tổng chu kỳ. Giải pháp: xếp hạng + ánh xạ trên GPU (1 khối bitonic sort), không đồng bộ host trừ mỗi ~16 chunk để kiểm tra hủy. ~+0,1 điểm.
3. SHRINK SỚM HƠN: ceiling/stage2.tsv: pop2048 shrink@0,25 cho −5686,2 (42–50 s) so với @0,35 −5685,8 (54–56 s): chất lượng bằng nhau, nhanh hơn ~15% (80% thời gian deep nằm ở pha quần thể lớn 0–35%). Chỉ cần đổi QC_SHRINK_AT mặc định 0,35 → 0,25. ~+0,1 điểm.
4. Nhỏ: initial_state sinh spin/RNG trên CPU mỗi lần chạy (≈2,3 MB htod mỗi lần, ~5–8% tầng A); mem_get_info + cấp phát mỗi run.
- Tổng ước lượng nếu làm cả (1)+(2)+(3): ~+0,5 điểm (≈ x1,8 tốc độ chạm ngưỡng). Thứ tự làm: (3) ngay (đổi hằng số), (2) rồi (1) khi có cửa sổ đo GPU.

## opt36 KẾT QUẢ & opt37 TRIỂN KHAI (22:25 VN 14/11)
- opt36 (quần thể deep 4096 + follow 1), 62 phút: 16 attempt (2,6/10 phút), tốt nhất -5084,385, trung vị -5079,23, 3 khối thắng; ≤-5082: 4/16 (25%; opt35 22%). KHÔNG có lợi ích rõ so với opt35 (trung vị -5077,9) và nhịp job sâu giảm ~15% ⇒ HOÀN quần thể về 2048.
- opt37 = opt35 + QC_SHRINK_AT 0,25 (nhanh hơn ~15% deep, chất lượng bằng nhau theo stage2.tsv) + QS_FOLLOW 2 + NHẬT KÝ HIỆU CHUẨN (lease_run.rs): ghi /data/calib/calib.jsonl {nonce, điểm S0, cờ thăm dò (1/1000 salt bỏ qua bộ lọc S0), năng lượng từng tầng, deep?, offset} cho salt tới tầng B, mọi salt thăm dò và 1/10 phần còn lại (trần 250 MB), và /data/calib/family-<digest>.json (params + topology) để dựng lại instance offline và huấn luyện bộ dự báo trên chính họ live (thay cho 300 instance harness). Hướng: dùng dữ liệu live thật như bản fork screen cũ (cal_*.csv) để hiệu chỉnh S0/q/tầng.
- Bản vá: opt37.patch. Quay về: ln -sfn /data/quip-cuda-otsuka-opt35.

## opt38 (đã build 22:32, CHƯA triển khai — chờ lần khởi động lại kế): LƯU BỀN TRẠNG THÁI SÀNG QUA KHỞI ĐỘNG LẠI
- lease_run.rs: cửa sổ xếp hạng các tầng A/B/C (self.screen), cửa sổ điểm S0 và q của S0 được ghi vào /data/calib/screen_state.json mỗi ≥30 s (ghi nguyên tử) và nạp lại lúc khởi động nếu số nút topology khớp. S0 đổi từ 'mỗi lease một cửa sổ' sang tĩnh toàn cục (hết 64 salt qua tự do ở đầu mỗi lease). Lý do: cửa sổ tầng C cần ~40 phút mới phản ánh thực tế (≈0,1 salt/s vào C), mỗi lần khởi động lại sàng lệch theo prior (prior A -4795 lệch: dữ liệu live cho A của các salt qua tầng ≈ -4822…-4847).
- KHÔNG làm được: tiếp tục job GPU dang dở (deep) sau khởi động lại — salt/nonce gắn với lease và round; sau restart coordinator cấp lại lease từ salt 1 nên nonce cũ không còn index để push. Việc có thể làm tương tự 'pool' của fork screen: lưu bền STASH của coordinator (ứng viên chưa đạt ngưỡng nhưng có thể thắng khi ngưỡng rã; ~2 trong ~30 khối thắng của ta đến từ stash) qua khởi động lại trong cùng round — ước ~80 dòng trong coordinator (stash.rs/runtime.rs); lợi ích nhỏ vì ít restart; chưa làm.
- Nhật ký hiệu chuẩn opt37 đã hoạt động: 128 dòng sau ~5 phút (dòng: nonce, s0, ex, st[A,B,C,deep], deep, off).

## THỬ KERNEL 3 KHỐI/SM (23:00–23:05 VN 14/11; miner dừng ~2,5 + 1 + 0,8 phút) — KẾT QUẢ ÂM, đã hoàn tác mã
- Thay đổi thử: parent int32→uint16 (shared 9→7 B/nút), `__launch_bounds__(512,3)`, QC_THREADS (block_dim 512/256). Dữ liệu: ceiling/occ.tsv, occ2.tsv, occ3.tsv.
- n=4800 (harness): parent int16 + 512 thread: bit-giống (cùng năng lượng) nhưng +3–6% chậm (0,90 vs 0,85 s B; 3,36 vs 3,26 C; 5,0 vs 4,7 P2048) — vẫn 2 khối/SM (3×(7·4800+1 KB) > 100 KB). 256 thread/khối: chậm ×1,57 ⇒ kernel CÓ phụ thuộc số thread/khối (ghi chú cũ 'threads/block không đổi' sai với kernel hiện tại).
- n=4048 (3 khối/SM vừa shared): launch_bounds(512,3): B256 0,76 vs 0,72 s (+5%), C512 2,87 vs 2,77 (+4%), P2048x16384 3,75 vs 4,03 s (−7%). Chỉ nhanh hơn ở quần thể lớn (pha 0–25% của deep), chậm hơn ở tầng B/C (≤512 khối) ⇒ tổng ≈ 0 hoặc âm. Số khối độc lập thứ 3 mỗi SM chỉ đem +7% (suy luận 'bị giới hạn bởi độ trễ' từ 1→2 khối/SM KHÔNG ngoại suy được lên 3). Hạng mục #1 trong rà soát mã: ĐÃ ĐO, ÂM — không triển khai. Mã kernel/capacity/device đã khôi phục về opt37.
- Còn lại từ rà soát: PA trên GPU (~7%), dùng lại đồ thị giữa tầng (1–2%), trạng thái ban đầu trên GPU (2–4%).

## opt39 TRIỂN KHAI 23:32 VN 14/11 (người dùng nhận xét opt36 có nonce sâu hơn opt37): quay lại cấu hình opt36 + lưu bền trạng thái sàng
- Bằng chứng (yếu, chưa đạt ý nghĩa thống kê): opt36 (21:22–22:25, quần thể 4096, shrink 0,35, follow 1): 16 attempt, 3 khối thắng (#344 -5084,385; #346 -5082,565; #348 -5084,025), ≤-5084: 2/16, trung vị -5079,2. opt37 (22:25–, quần thể 2048, shrink 0,25, follow 2): 13 attempt/41 phút, trung vị -5078,2, tốt nhất -5082,335, 1 khối thắng (#350 -5082,335), ≤-5083: 0/13. Trung vị các bản đều trong ±1,5 (sai số ±1) nhưng đuôi sâu của opt36 tốt hơn, và dữ liệu stage2 gợi ý quần thể lớn sâu hơn; opt37 đổi 3 thứ cùng lúc (pop, shrink, follow) nên không tách được nguyên nhân.
- opt39 = opt36 (CUDA_ADAPT reads 4096, QC_SHRINK_AT 0,35, QS_FOLLOW 1) + opt38 (lưu bền cửa sổ sàng/S0 trong /data/calib/screen_state.json) + nhật ký hiệu chuẩn. Bản vá: opt39.patch. Chuỗi 6 khối liên tiếp không thắng lúc triển khai (đã vượt ngưỡng 5).
- Đánh giá: ~20 job sâu (~2 giờ vì job dài 100 s); so đuôi ≤-5083 và số khối thắng với opt36/opt37.

## A/B DEEP ĐO TRỰC TIẾP (23:36–23:49 VN 14/11, miner dừng 13 phút theo quyền tự chủ khi thua 6 khối): KẾT LUẬN — cấu hình gốc 2048/shrink 0,35 là tốt nhất
- Harness, 65536 sweep, 3 instance (0,3,1) x seed (11,22) = 6 mẫu/cấu hình (ceiling/deepab.tsv): p2048_s35 −5689,06 (22,8 s); p2048_s25 −5687,38 (18,4 s); p4096_s35 −5687,39 (41,0 s); p4096_s25 −5688,54 (31,8 s). Chênh lệch theo cặp so với p2048_s35: s25 +1,68 (sd 2,9); p4096 +1,68 (sd 3,3); p4096_s25 +0,53 (n=6, sai số ±1,2 harness ≈ ±0,8 live).
- ⇒ quần thể 4096 KHÔNG sâu hơn 2048 (gấp đôi thời gian, không lợi); shrink 0,25 mất ~1,2 live để nhanh hơn 19% (dữ liệu stage2 cũ 'bằng nhau' sai do mẫu nhỏ) — không đáng. Nhận xét 'opt36 sâu hơn opt37' là biến động mẫu (3 khối thắng/62 phút của opt36 là may mắn của đuôi + nhịp).
- opt40 TRIỂN KHAI 23:50 (chuỗi 6 khối không thắng): quần thể 2048, shrink 0,35, QS_FOLLOW 2 (= cấu hình opt35) + lưu bền cửa sổ sàng + nhật ký hiệu chuẩn. Bản vá opt40.patch. Quay về: ln -sfn /data/quip-cuda-otsuka-opt35 rồi docker restart quip-cuda.

## opt41 TRIỂN KHAI 00:03 VN 15/11 (chuỗi 8 khối không thắng; người dùng: đối thủ TB -5083, cao nhất -5087): SÀNG CHỌN LỌC HƠN
- Bối cảnh: ngưỡng round ~ -5082,7…-5083,5 (đối thủ thắng sát ngưỡng -5082,2…-5082,8, thỉnh thoảng -5084,9/-5086,7). Phân bố job sâu của ta (TB -5079, sd ~3,25) cho P(≤-5083) ≈ 10%, P(≤-5085) ≈ 3% mỗi job. opt39/opt40 KHÔNG đổi chất lượng (A/B chứng minh quần thể 4096 không hơn) ⇒ không tự tin thắng nhiều hơn opt35; kỳ vọng chỉ vài khối/đêm.
- sim_confirm (mô hình cửa sổ đúng, 40 giờ/cấu hình): (qS0 64, qA 32, qB 16, qC 4, follow 2): P(≤-5083) 1,40/giờ vs live 1,00; P(≤-5085) 0,375 vs 0,175 (x2,1); trung vị -5081,0 vs -5080,0; job sâu 9/giờ vs 14,7; nguồn 663 salt/s (khả thi: lease đã đo 625–756 salt/s). Chọn đuôi sâu thay vì số lượng vì ngưỡng cao.
- Thay đổi: QS_STAGES mặc định 256:64:16;256:8192:8;512:16384:4 → 256:64:32;256:8192:16;512:16384:4 (opt40 + q sàng). Bản vá opt41.patch. Đánh giá sau ~2 giờ (≥ ~15 job sâu): trung vị kỳ vọng ~-5081; quay về opt40: ln -sfn /data/quip-cuda-otsuka-opt40 + docker restart quip-cuda.

## PHÂN TÍCH NHẬT KÝ HIỆU CHUẨN LIVE (00:40 VN 15/11; 3136 dòng, topology 4560 nút; công cụ: generation_cost.rs::calib_features dựng lại instance từ nonce + family-<digest>.json)
- Dựng lại thành công 3136 instance live từ nonce. Đặc trưng live KHÁC harness (mật độ f0 0,035 vs 0,0177) nhưng điểm S0 dùng xếp hạng cửa sổ nên vẫn hoạt động.
- Nhãn không thiên lệch (830 dòng thăm dò, bỏ qua S0): corr(S0 hiện tại, năng lượng tầng A) = 0,305; ridge huấn luyện lại trên đặc trưng live CV 0,29; train-trên-không-thăm-dò→test-thăm-dò 0,325; GBR kém hơn (0,19–0,36). ⇒ KHÔNG có cải thiện đáng kể từ huấn luyện lại (+0,02): S0 hiện tại đã gần tối đa của lớp đặc trưng đếm tam giác (ρ≈0,3 với A).
- Tương quan trong tập vượt qua (range-restricted): corr(A,B)=0,09; corr(B,C)=0,19. Năng lượng A: TB −4843,2 sd 9,7; salt qua A TB −4855,8.
- Phát hiện vận hành: thăm dò 1/1000 chiếm ~5% đầu vào phễu (A chỉ ~3–10 salt/s) ⇒ pha loãng lợi ích S0 (~−0,15 điểm). Đã đủ 830 nhãn ⇒ giảm thăm dò xuống 1/20000.

## opt41 ĐÁNH GIÁ 60 PHÚT (01:03 VN 15/11) + LỖI BỘ ĐIỀU KHIỂN S0 → opt42 (đã build, CHƯA triển khai: chuỗi không thắng = 0 vì ta vừa thắng #364 và #366)
- opt41 (sàng q 32/16/4, từ 00:03): 10 attempt/60 phút (1,7/10 phút), TRUNG VỊ -5081,71 (opt40 ~-5078…-5079; mô phỏng dự đoán -5081,0 ✓), tốt nhất -5083,955; ≤-5083: 2/10; 2 KHỐI THẮNG: #364 00:36:47 (-5083,955, thr -5083,575) và #366 01:01:28 (-5082,115, thr -5082,109). Đuôi chọn lọc hơn đúng như mô phỏng. GIỮ opt41 làm bản nền.
- LỖI THIẾT KẾ tìm thấy khi đo GPU (56% util, đợt rảnh 1–2 s lặp lại, CPU container ~750%): S0::adapt thắt q ×1,03 MỖI salt nhận ngay (tín hiệu 'tồn đọng' sai) nhưng chỉ nới q ×0,97 mỗi 20 ms timeout ⇒ cân bằng cấu trúc ở ~39% GPU rảnh (1,52 f = 0,98 (1−f)). opt35 (99%) không bị vì ít salt tầng A hơn mỗi job sâu.
- opt42 = opt41 + (a) bộ điều khiển theo độ dài hàng đợi (AtomicUsize queued): left≥5 thắt ×1,04; left==0 nới ×0,96; timeout 20 ms nới ×0,85 ⇒ GPU luôn có việc với bộ lọc chặt nhất có thể; (b) thăm dò S0 1/1000 → 1/20000 (đã đủ 830 nhãn; thăm dò chiếm ~5% đầu vào phễu). Kỳ vọng: GPU ≥ 95%, +20–40% thông lượng tầng A. Bản vá opt42-lease-run.patch. Triển khai khi chuỗi không thắng ≥ 5. Quay về: ln -sfn /data/quip-cuda-otsuka-opt41.

## DASHBOARD: KHỐI ĐÃ NỘP THÀNH CÔNG BỊ THIẾU / HIỆN 'rejected' (01:50 VN 15/11) — đã vá dữ liệu cũ
- Đối chiếu chain (/api/mining-history, 69 khối thắng của ta từ 13/11) với attempts: sau khi coordinator đã vá được triển khai (21:22 VN 14/11) CẢ 8 khối thắng (#344, #346, #348, #350, #364, #366, #368, #371) đều có bản ghi submitted=true kèm chain_block_number ✓. Các khối thiếu bản ghi nộp nằm ở thời kỳ coordinator rc10 (đường win-time không ghi attempt): #303, #309, #310, #333 (và 13 khối 13/11 #99–#132 từ trước khi có nhật ký attempts, không khôi phục được năng lượng).
- Vá dữ liệu: cập nhật 4 dòng attempts (accepted/submitted=true, chain_block_number=13988/14520/14623/16338, best_energy=raw) trong dir 302/308/309/332; bản gốc sao lưu attempts-backfill/. Job: f02d4bec (-5080,765), 0ce03c3e (-5079,615), 88bd2f06 (-5079,455), ac8d8cfb (-5080,475).
- Nguyên nhân gốc đã sửa từ trước trong coordinator 0.3.9-rc5 vá (AttemptRecord::win_time).

## GIẢ THUYẾT '+1' VỀ KHỐI NỘP SỚM (01:55 VN 15/11, người dùng): ĐÃ KIỂM TRA + CỨNG HÓA coordinator
- Kiểm tra số liệu: 52 khối thắng của ta khớp chain_block_number == substrateBlockNumber; 8 khối sau bản vá nằm đúng thư mục qblock−1 (dashboard #N ↔ thư mục N−1) ⇒ KHÔNG có lệch +1 trong dữ liệu hiện có. 4 khối thiếu (#303, #309, #310, #333) do thiếu bản ghi nộp ở đường win-time của rc10 (đã vá dữ liệu).
- Rủi ro cùng loại có thật về mặt cấu trúc: biên lai nộp trả về lúc khối được đưa vào chain, đúng lúc round kế bắt đầu; đọc st.qblock_id SAU biên lai có thể ghi bản ghi vào thư mục qblock+1. Đã cứng hóa trong coordinator: chốt qblock_at_submit TRƯỚC khi gọi submit_proof (session.rs settle_pow_result và runtime.rs win-time). Bản vá: coordinator-wintime-attempt.patch (cây rc9, 334 test qua) và coordinator-rc5-patched2.patch (cây v0.3.9-rc5); binary /data/quip-coordinator-0.3.9-rc5-patched2 (CHƯA triển khai; thay khi cần restart, quay lại: patched).

## opt42 TRIỂN KHAI 02:06 VN 15/11 (người dùng yêu cầu; chuỗi 3 khối không thắng; GPU chỉ 51%, ngưỡng round -5085,5)
- opt42 = opt41 + bộ điều khiển S0 theo độ dài hàng đợi (left≥5 thắt ×1,04; left==0 nới ×0,96; timeout 20 ms nới ×0,85) + thăm dò S0 1/20000. Sửa lỗi cân bằng cấu trúc ~39% GPU rảnh của opt41. Bản vá opt42-lease-run.patch. Mốc đánh giá: mức dùng GPU (opt41 51–57%), job sâu/giờ, trung vị.
- Quay về: ln -sfn /data/quip-cuda-otsuka-opt41 rồi docker restart quip-cuda. Coordinator đã củng cố (patched2) CHƯA triển khai (tránh nhiễu đánh giá).

## opt43 TRIỂN KHAI 02:21 VN 15/11 (người dùng yêu cầu): GIẢM CHI PHÍ HOST CỦA LUỒNG GPU + ĐO PHA
- Đo opt42 (02:14): GPU 60% (opt41 51–57%), 25% mẫu <30%, 28 đợt rảnh/90 s (dài nhất 2,2 s); luồng điều khiển GPU 99,9% CPU trong khi 24 luồng sinh gần như rảnh (CPU máy còn ~67% trống) ⇒ nghẽn ở việc phía host mỗi salt (ước ~15 ms/salt tầng A: sinh trạng thái ngẫu nhiên đầu, tải mô hình, cấp phát, tải 16 read, unpermute + tính lại năng lượng).
- opt43 = opt42 + (a) các tầng sàng A/B/C chỉ tải 1 read tốt nhất (thread-local KEEP_OVERRIDE trong device.rs; deep vẫn 16) vì salt bị loại chỉ dùng screen(); (b) bộ đo thời gian host theo pha (PROF_US: init, launch_sync, post, work_new, run_total, score) ghi log mỗi phút 'device thread host-side time per minute' để chọn bước tiếp (vd sinh trạng thái đầu trên GPU, tái dùng ClusterDevice giữa các tầng).
- Mốc đánh giá: mức dùng GPU (opt42 60%), log pha. Quay về: ln -sfn /data/quip-cuda-otsuka-opt42 rồi docker restart quip-cuda. Bản vá opt43.patch.

## GPU RẢNH: GIẢ THUYẾT 'LỌC QUÁ CHẶT → ÍT JOB' + RECALL (02:35 VN 15/11)
- Log pha opt43 (mỗi phút): 947–1320 salt/phút (~20/s), luồng GPU bận 92% (init 4–6 s, launch_sync 46–50 s, post ~1 s) ⇒ KHÔNG đói việc; GPU rảnh nằm bên trong run (bước PA đồng bộ host ở tầng B/C/deep). Mức dùng GPU: opt41 51–57% → opt42 60% → opt43 74% (mẫu<30%: 30% → 25% → 9%).
- Recall (recall2.py, 16 triệu instance, ρ S0 0,15 bảo thủ, A 0,39, B 0,61, C 0,70, một dòng dõi deep): bộ lọc chặt giữ rất ít instance tốt (qS0 45, 32/16/4: 0,7–2,5% instance F≤-5083,5 sống sót; (16,16,8,4): 6%), NHƯNG hiệu suất = instance tốt tới deep trên 1000 GPU-giây gần phẳng: @-5083,5: (16,16,8,4) 0,15; (16,32,16,4) 0,14; (45,16,8,4) 0,14; (45,32,16,4) 0,12; (45,32,16,8) 0,15; @-5085: chặt hơn tốt hơn (0,062–0,073 vs 0,025–0,033 ở lọc lỏng). ⇒ chặt không 'làm mất' hiệu suất theo GPU-giây; GPU tiết kiệm được dùng để đánh giá thêm instance. Chi phí thật của lọc chặt là độ trễ (mỗi job sâu ~3 phút) khi round ngắn — chờ sim_round.

## sim_follow2 XONG (03:35 VN 15/11; 12 giờ mô phỏng/ô; stages (64,32,16,4); mục tiêu exit -5084,7)
- promise -5081: follow 1 → deep/h 11,8 P83 2,50 P85 0,917; follow 2 → 10,7 / 2,25 / 0,917; follow 4 → 9,4 / 2,25 / 0,833; follow 6 → 8,2 / 2,00 / 0,667. promise -5079: follow 1 P85 0,750 (2,33), follow 2 0,667, follow 4 0,667, follow 6 0,750. promise -5074 (opt41–43): follow 1 0,750; follow 2 0,583 (P83 2,08); follow 4 0,583; follow 6 0,500.
- KẾT LUẬN: chỉ chạy dòng dõi phụ khi dòng dõi đầu ≤ -5081 (thay vì ≤ -5074) tăng P(≤-5085) +57% (0,583→0,917) và P(≤-5083) +20%, đồng thời tăng 30% job sâu/giờ (dòng dõi phụ cho instance trung bình là lãng phí). Số dòng dõi phụ (1–2) gần như không đổi ở promise -5081. opt44 = opt43 + QS_PROMISE mặc định -5081 (build, chưa triển khai: chuỗi không thắng 1, ta thắng #382 và #384). Triển khai khi chuỗi ≥5 (hoặc người dùng yêu cầu).
- opt43 72 phút: 12 attempt, trung vị -5079,04 (hồi phục so với -5077,4 ở 42 phút ⇒ lần 'giảm' là nhiễu), tốt nhất -5084,885; 2 khối thắng (#382 -5083,015; #384 -5084,885).

## opt44 + COORDINATOR patched2 TRIỂN KHAI 04:44 VN 15/11 (chuỗi 5 khối không thắng; người dùng: 'opt43 khá tệ, nhiều nonce -507x')
- So sánh phân bố attempt (trung vị | ≤-5081 | ≤-5083 | >-5078): opt32 46 mẫu -5079,81|30%|11%|15%; opt35 9 -5077,86|33%|11%|56%; opt36 16 -5079,23|25%|12%|19%; opt37 21 -5079,70|24%|0%|33%; opt41 15 -5081,60|67%|20%|13%; opt43 23 -5079,32|39%|17%|30%. opt41 là ngoại lệ tốt nhất (p≈2% nếu cùng phân bố opt43; 6 phiên bản ⇒ có thể là 'lời nguyền người thắng'); opt43 ≈ mức chung của các bản trước. Giả thuyết bộ điều khiển S0 (opt42/43, q thấp hơn) làm xấu mẫu: ảnh hưởng ước tính ≤ 0,3–0,5 điểm (ρ(S0,F) 0,15–0,3) < chênh lệch quan sát 2,3 ⇒ chủ yếu nhiễu; chưa đảo ngược.
- opt44 = opt43 + QS_PROMISE -5081 (sim_follow2: P(≤-5085)/h +57%, P(≤-5083) +20%, job sâu +30%; chỉ chạy dòng dõi phụ khi dòng đầu ≤ -5081). Coordinator 0.3.9-rc5-patched2 (chốt qblock_at_submit trước khi nộp) cùng đợt khởi động lại. Quay về: ln -sfn /data/quip-cuda-otsuka-opt43 (miner) và /data/quip-coordinator-0.3.9-rc5-patched (coordinator) rồi docker restart quip-cuda.
- Đánh giá ~2 giờ: tỉ lệ ≤-5081 (opt43 39%) và >-5078 (30%), trung vị, số khối thắng.

## CHỈ THỊ NGƯỜI DÙNG (04:46 VN 15/11): NGƯỠNG TỐI ƯU ĐỔI TỪ 5 XUỐNG 3 KHỐI THUA LIÊN TIẾP
- Khi ta thua ≥3 khối liên tiếp (người khác thắng): được tự tối ưu, dừng miner đo GPU, build và triển khai bản thử/hoàn tác. Chuỗi <3 (vừa thắng): chỉ giám sát, không restart/deploy.

## sim_round (theo độ dài round) KHÔNG XONG: bị `timeout 7200` hủy ở 2000/4000 (không in kết quả vì chỉ in ở cuối). Chạy lại sim_round2.py (lưới nhỏ: 5 cấu hình x 8 độ dài round x 10 seed, in kết quả tạm mỗi 40 job, timeout 4 giờ). Bài học: mọi mô phỏng dài phải in kết quả tạm theo từng đợt.

## sim_round2 (05:05 VN 15/11) — KẾT QUẢ KHÔNG DÙNG ĐƯỢC (lỗi mô hình)
- Mô phỏng khởi động lại cửa sổ xếp hạng RỖNG ở mỗi round ⇒ các cửa sổ cần ≥24 mẫu/tầng mới cho qua; cấu hình (64,32,16,4) cho 0,02 job sâu/round (thực tế opt41–44: 1,6–2,6 job sâu/10 phút, ~1–2 mỗi round) vì trên miner thật cửa sổ được GIỮ giữa các round (và nạp lại từ screen_state.json). Các cấu hình lỏng (0,6–0,8 job sâu/round) cũng bị đánh giá thấp. Bỏ qua kết quả sim_round2; cần khởi tạo cửa sổ 'ấm' nếu mô phỏng lại.
- opt44 19 phút đầu (04:44–05:03): 5 attempt (2,6/10 phút), trung vị -5082,94, tốt nhất -5084,005, 0 nonce >-5078, ≤-5081 3/5; 1 khối thắng #397 (04:54, -5084,005, thr -5083,968). Mẫu nhỏ nhưng tốt.

## ĐO CHI PHÍ BƯỚC PA ĐỒNG BỘ HOST (06:03–06:05 VN 15/11, miner dừng ~1,5 phút; chuỗi 3 khối thua) — KẾT QUẢ ÂM: PA-trên-GPU không đáng
- Harness n=4048 nút, instance 0, 2 seed (ceiling/pa_cost.tsv). Thời gian chạy: B 256x8192: PA32 0,795/0,808 s, PA tắt 0,764/0,747, PA64 0,760/0,757 ⇒ overhead PA32 ~+5–7%; C 512x16384: 2,996/3,035 vs 2,950/2,944 vs 2,946/2,977 ⇒ ~+2%; D 2048x16384 (có shrink): PA32 5,40 s ≈ PA64 5,34 s (PA tắt 11,6 s không so được vì shrink nằm trong nhánh PA) ⇒ ~+1%. ~0,16 ms/bước.
- Chuyển xếp hạng PA sang GPU chỉ tiết kiệm ≲2–3% tổng chu kỳ (B chiếm ~30% thời gian sàng nhưng overhead chỉ 5–7%). PA mỗi 64 gần như không nhanh hơn và năng lượng kém nhẹ (B -4723/-4718 vs -4732/-4725). Hạng mục 'PA trên GPU' ĐÃ ĐO, ÂM — không làm. Ước lượng cũ (~7%) quá lạc quan: kernel anneal lớn hơn nhiều so với bước host.
- opt44 79 phút: 12 attempt, trung vị -5081,73, tốt nhất -5085,085 (#400), 2 khối thắng (#397 -5084,005; #400 -5085,085).

## ĐÁNH GIÁ LẠI THEO YÊU CẦU NGƯỜI DÙNG (09:40 VN 15/11; chuỗi thua 2, trước đó 7)
- Người dùng: trung vị opt44 tệ hơn opt35/36/37; 3 khối thắng có thể là may; thuật toán mới chỉ là lọc nonce; yêu cầu nghĩ lại / viết lại thuật toán.
- Số liệu (từ ghi chú): trung vị opt35 -5077,9 (9 mẫu), opt36 -5079,2, opt37 -5079,7, opt32 -5079,8, opt44 -5079,9 (57 mẫu) ⇒ opt44 KHÔNG tệ hơn về trung vị. Chỉ số có ý nghĩa hơn là số nonce ≤-5083 trên GIỜ: opt32 ≈1,4/h (11% x 2,1 job/10 phút), opt44 ≈2,5/h (12 trong 289 phút). Số khối thắng không so được giữa các thời điểm vì ngưỡng đã tăng từ -5080 lên -5085.
- Thừa nhận: đúng, từ opt17 phần lõi giải KHÔNG đổi; mọi cải thiện sau đó là sàng (lọc instance), thông lượng, dòng dõi phụ. Bộ giải cụm Otsuka đã bão hòa theo mọi trục đã thử (xem bảng KHÔNG HIỆU QUẢ: sweeps, quần thể, PT, reheat, crossover, worm, DP chính xác, kernel CPU dimer).
- Tra tài liệu upstream quip-solver-cpu/docs/comparisons.md: cpu-bsb (Simulated Bifurcation) KÉM hơn cpu-sa ở mọi cỡ 0,10–0,56%; cpu-sb chỉ hơn sa 0,05–0,17% ở cỡ nhỏ ⇒ viết lại bằng SB gần như chắc chắn thua cụm Otsuka (0,1% của -5080 ≈ 5 đơn vị theo hướng XẤU). Không đáng viết prototype GPU.

## ĐO PHÂN TÁN THEO SEED TRÊN CÙNG MỘT INSTANCE (09:59–10:13 VN 15/11; miner dừng 14 phút, người dùng duyệt)
- Harness instance 0, deep thật (2048 reads x 131072 sweeps, PA 32/0,3, shrink 0,35→256, beta 1,2–8), 16 seed (101–116), ~47,7 s/lần (GPU ép xung +150 MHz, SM 2880 MHz).
- Kết quả: min -5695,86; trung vị -5687,87; trung bình -5688,79; sd 3,13; max -5684,75. Tốt nhất cách trung vị 8,0 đơn vị. (ceiling/seedspread.tsv)
- Ý nghĩa: sd theo seed trên MỘT instance (3,1) ≈ sd của kết quả deep live (3,25) ⇒ biến thiên lớn nhất là 'bồn' mà quần thể sụp vào, không phải instance. Bộ giải còn xa trạng thái cơ bản (≥8 đơn vị trên instance này). Bão hòa theo sweeps/pop nhưng KHÔNG bão hòa theo số lần khởi động độc lập: tỉ lệ ≤ trung vị-8 là 1/16.
- Kỳ vọng best-of-k (sd 3,1): k=2 +1,8; k=4 +3,2; k=8 +4,3 (đơn vị) so với 1 lần. Chi phí tuyến tính ⇒ cần cách chọn bồn sớm rẻ hơn chạy hết (race-abort ở 25% đã có; đo xem năng lượng tại checkpoint dự báo cuối bao nhiêu).
- Việc kế: ghi năng lượng tại các checkpoint (4096/16384/32768 sweeps) cho 16 seed để tính tương quan với kết quả cuối ⇒ quyết định chiến lược nhiều lần khởi động ngắn + chọn lọc sớm (successive halving) thay vì 1 deep.

## TƯƠNG QUAN NĂNG LƯỢNG GIỮA CHỪNG ↔ KẾT QUẢ CUỐI (10:19–10:33 VN 15/11; miner dừng 14 phút) — KẾT QUẢ ÂM cho 'chọn lọc dần'
- Harness instance 0, deep thật, 16 seed 101–116, QC_TRACE=1 (năng lượng tốt nhất quần thể mỗi 32 sweep). Kết quả cuối lần này: min -5695,79, trung vị -5690,34, sd 2,27 (lần đo trước sd 3,13; cùng seed KHÔNG tái lập bit-giống nhau giữa hai lần chạy).
- Tương quan Pearson (năng lượng tốt nhất tại mốc vs kết quả cuối; n=16): 1024: 0,28; 2048: -0,11; 4096: 0,35; 8192: 0,60; 16384 (12,5%): -0,54; 32768 (25%): 0,56; 45888 (35%, lúc thu nhỏ): 0,77; 65536 (50%): 0,88; 98304 (75%): 0,95.
- Kết luận: không có bộ dự báo sớm đáng tin trước ~35% lịch (mốc race-abort 25% chỉ r≈0,56). Kết quả được quyết định quanh lúc quần thể sụp. Chi phí tới 35%: 0,35x2048 = 717 đơn vị so với 0,65x256 = 166 phía sau ⇒ ~81% thời gian nằm TRƯỚC mốc dự báo được ⇒ bỏ sớm ở 35% chỉ tiết kiệm ≤19%. 'Nhiều lần khởi động + successive halving' KHÔNG khả thi; best-of-k phải trả đủ k lần chi phí (~81% mỗi lần).
- Ghi chú: phân tán theo seed lớn (sd 2,3–3,1, tốt nhất hơn trung vị 5–8) nhưng không có cách rẻ để chọn bồn sớm ⇒ lợi ích nằm ở thuật toán khiến quần thể KHÔNG sụp vào bồn ngẫu nhiên (đa dạng sau 35%), mà islands/crossover đã thử không giúp. Hướng chưa thử còn lại: sụp theo từng nhóm (giữ nhiều bồn 4x64 thay vì 1x256 sau 35% rồi chọn ở 60–75% khi r>0,9) — chi phí sau 35% chỉ 166 đơn vị nên chạy 4 nhóm sau điểm sụp rẻ (~+16% tổng).

## THỬ NGHIỆM 4 NHÓM x 64 SAU THU NHỎ, CHỌN NHÓM TỐT NHẤT Ở 60% (10:45–10:59 VN 15/11) — KẾT QUẢ: KHÔNG CÓ Ý NGHĨA, KHÔNG TRIỂN KHAI
- Mã: device.rs, env QC_ISL2 (số read mỗi nhóm sau shrink; PA chỉ trong nhóm; hàng sống sót chia vòng tròn giữa các nhóm) + QC_PICK_AT (giữ nhóm có năng lượng tốt nhất ở mốc này). Build harness cần `--lib --features long-tests` (test measure_generation_cost nằm sau feature này).
- Harness instance 0, 16 seed 201–216, QC_ISL2=64 QC_PICK_AT=0.6: trung vị -5690,55, trung bình -5690,14, sd 3,15, tốt nhất -5694,58; thời gian 46,5 s (không đổi). Bản gốc (32 lần chạy: seedspread+seedtrace): trung vị -5689,28, trung bình -5689,48, sd 2,78, tốt nhất -5695,86. Mann-Whitney p=0,19; P(≤ trung vị gốc -5) bằng nhau 0,06.
- Kết luận: +0,7…+1,3 ở phần giữa nhưng KHÔNG ở đuôi (thứ cần để thắng) và không đạt ý nghĩa; muốn xác nhận +1 cần ≥64 mẫu/nhánh (~55 phút dừng miner). Mã giữ nguyên (mặc định tắt, QC_ISL2=0 ⇒ hành vi cũ). KHÔNG deploy.
- Tổng thời gian miner dừng cho đo hôm nay: ~42 phút (seed 14 + trace 14 + 4x64 14), cộng ~3 lần script lỗi (ngắn).

## JOB SÂU BỊ HỦY KHI ĐỔI ROUND KHÔNG CÓ NHẬT KÝ (11:10 VN 15/11; người dùng chỉ ra: 'lỗi từ đầu')
- Xác nhận: khi đối thủ thắng, coordinator hủy job đang chạy (log 'cancelled_jobs=N'; 15/58 round trong 8 giờ có ≥1 job bị hủy). Job sâu đang chạy dở bị bỏ: sampler trả lỗi hủy, lớp upstream loại kết quả theo watermark ⇒ không có dòng nào trong attempts.jsonl hay dashboard (chỉ job sâu hoàn tất, hoặc dòng dõi đầu hoàn tất khi dòng phụ bị hủy, mới được ghi).
- Mất mát thực: ước ~3% thời gian GPU (job sâu 52 s, round ~10 phút, xác suất đang ở deep ~65%); năng lượng dở dang thuộc round cũ nên KHÔNG nộp được ⇒ không ảnh hưởng số khối thắng, chỉ thiếu nhật ký.
- Sửa phía miner (opt45, CHƯA triển khai, gộp với đợt triển khai sau): device.rs lưu tiến độ deep (DEEP_PROGRESS: sweep đã chạy/tổng/năng lượng tốt nhất mỗi 32 sweep) và take_deep_progress(); lease_run.rs ghi dòng {"cancelled":1,"ts_ms","done","total","best_milli"} vào /data/calib/calib.jsonl khi job sâu bị hủy (xóa tiến độ cũ khi job hoàn tất). Bản vá: opt45-cancelled-log-and-isl2.patch (gồm cả QC_ISL2 mặc định tắt). Build OK; chưa chạy thử trên miner.
- Chưa làm: đưa dòng 'rejected' vào attempts.jsonl của coordinator (cần miner báo job bị hủy kèm năng lượng; hiện lớp upstream loại bỏ kết quả hủy).

## opt45 TRIỂN KHAI 11:13 VN 15/11 (người dùng: dashboard Recent Performance thiếu khối bị từ chối khi round bị hủy)
- opt45 = opt44 + ghi dòng {"cancelled":1,...} vào calib.jsonl khi job sâu bị hủy (xem mục trước) + QC_ISL2 (mặc định tắt). Nonce/thuật toán KHÔNG đổi. Restart ngay sau round 441 (80 s). Quay về: ln -sfn /data/quip-cuda-otsuka-opt44 rồi docker restart quip-cuda.
- Script phụ phía host `scratch/cancelled_to_attempts.py --loop` (mỗi 60 s): đọc dòng 'cancelled' mới, gán vào round đang chạy trước thời điểm hủy (qua 'new round' trong log coordinator), thêm một dòng attempt accepted=false submitted=false (raw_best_energy_milli = năng lượng tốt nhất lúc bị hủy, device_access_time_us ước = tiến độ x 52 s) vào attempts/<qblock>/attempts.jsonl ⇒ dashboard hiển thị 'rejected'. Cùng cơ chế với backfill trước đây. Chưa kiểm chứng đầu-cuối: chờ lần hủy tiếp theo.

## opt46 (11:15 VN 15/11; người dùng: nhật ký bị nhảy khối, vd 429→436) — ĐÃ BUILD, tự triển khai ngay sau round kế (deploy46.sh)
- Nguyên nhân khoảng trống: round ngắn (≈2 phút: 429, 430, 431) hoặc coordinator khởi động lại (433, 434, 437, 438, 440–442) không có job sâu nào hoàn tất ⇒ attempts/<q>/ không tồn tại.
- opt46 = opt45 + tóm tắt sàng mỗi round: bộ đếm SCR_BEST/SCR_N (năng lượng tốt nhất và số salt đã sàng, không deep) ghi cùng dòng 'cancelled' (trường scr_best_milli, scr_n) mỗi khi round đổi; script phụ cancelled_to_attempts.py lấy min(năng lượng deep dở dang, năng lượng sàng tốt nhất) làm raw_best_energy của dòng 'rejected' ⇒ mỗi round có miner làm việc đều có ≥1 dòng. Năng lượng sàng yếu (ngắn, ~-5050…-5065) và device_access_time_us=0 khi không có deep ⇒ phân biệt được.
- Còn khoảng trống khi miner/coordinator khởi động lại (không ai ghi gì). Quay về: opt44.

## opt47 TRIỂN KHAI 11:22 VN 15/11 (gợi ý của người dùng: tách GPU, không để GPU rảnh) — 2 WORKER GPU CHẠY ĐỒNG THỜI
- Cơ sở: GPU util 72–82% (nvidia-smi) trong khi luồng GPU bận ~92% ⇒ ~20% thời gian GPU rảnh vì pha host (khởi tạo trạng thái 4–6 s/phút, sắp xếp PA mỗi 32 sweep ~5–7%, đọc về). Chọn 'dành SM cho deep, SM còn lại để sàng' (ý kiến người dùng) tương đương chạy hai luồng/stream đồng thời: khi một worker ở pha host, worker kia dùng GPU.
- opt47 = opt46 + QS_GPUW (mặc định 2, 1–4): vòng tiêu thụ GPU trong run_lease chạy bằng 2 worker dùng chung hàng đợi (Mutex<Receiver>); cửa sổ xếp hạng, S0, RACE_WINDOW dùng chung (đã có mutex/atomic); DEEP_PROGRESS thread-local. Quay về: QS_GPUW=1 không có sẵn qua env trong container ⇒ ln -sfn /data/quip-cuda-otsuka-opt46 rồi docker restart.
- Mốc đánh giá (≥40 phút): mức dùng GPU (opt44 72–82%), số job sâu/10 phút (opt44 ~1,8), lỗi/DeviceBusy/Capacity (skip salt), trung vị/đuôi nonce không được xấu đi (job chạy song song mỗi job lâu hơn ⇒ nhiều job bị hủy khi đổi round hơn; sẽ thấy qua dòng 'cancelled').
- 'Hủy sớm job xấu' (gợi ý đầu của người dùng): đã có 3 mức (S0 cấu trúc, 3 tầng sàng, race-abort 25%); đo tương quan cho thấy không có dự báo tin cậy trước 35% (r 0,56 ở 25%), hủy ở 50% (r 0,88) chỉ tiết kiệm ~4–7% tổng.

## 11:33 VN: opt47 10 phút đầu: THẮNG #442 (11:32, -5084,475, ngưỡng -5084,466). Phát hiện: calib_write chỉ xả bộ đệm mỗi 64 dòng ⇒ dòng 'cancelled' có thể trễ đến vài phút (round 443 bắt đầu với cancelled_jobs=2 nhưng calib.jsonl chưa có dòng). opt48 (đã build, CHƯA triển khai, chờ đợt gộp) xả ngay các dòng cancelled. Số đo opt47: GPU 85% (p10 76) so với 72–82% của opt44; run_total ~112–127 s/phút (2 worker ~95% bận).

## THỬ BỘ DỰ BÁO PHỔ (λ_min ma trận liên kết YẾU) — ÂM trên chain thật (11:50 VN 15/11; người dùng yêu cầu ý tưởng ngoài dự án)
- Cảm hứng: cận phổ cho năng lượng nền Ising. Trên 20 instance harness (đồ thị Zephyr): Spearman(λ_min J_yếu, năng lượng tầng B256) = 0,52 (p=0,02, n=20; mean-field của em bị tràn do liên kết mạnh -1000, bỏ).
- Kiểm chứng trên 806 instance chain thật (calib_snap, st≥2 tầng, CPU, 31 ms/instance gồm sinh + 300 vòng lũy thừa; test calib_features thêm trường lw): Spearman(lw, B) = -0,01…-0,02; kiểm chéo 5-fold: đặc trưng đếm F r=0,298; lw r=-0,07; F+lw r=0,295 (không cộng thêm). ⇒ 0,52 là nhiễu mẫu nhỏ/riêng đồ thị harness. Bộ dự báo phổ KHÔNG hữu ích. (Lưu ý: tập này đã qua S0 + tầng A nên tương quan bị co hẹp; so sánh công bằng: S0 score 0,23.)
- Mã: generation_cost.rs (calib_features thêm lw), scratch/predictor_test.py. Đừng lặp lại: dự báo chất lượng instance bằng đại lượng phổ/mean-field.

## opt47 sau 41 phút (12:03 VN 15/11) + sửa lỗi nhật ký 'hủy' (opt49, tự triển khai sau round kế)
- opt47 41 phút: 11 dòng attempt (2,7/10 phút so với opt44 1,8), tốt nhất -5084,475 (thắng #442), trung vị -5078,09, ≤-5081 4/11, >-5078 5/11; GPU 77% (p10 48) trong mẫu 20 s, run_total hai worker 90–113 s/phút (75–94% bận). Mẫu quá nhỏ để kết luận chất lượng nonce; số dòng attempt/10 phút tăng ~50%.
- Lỗi nhật ký: sau 40 phút calib.jsonl vẫn 0 dòng 'cancelled' dù có round đổi với cancelled_jobs=2 ⇒ lỗi hủy được ánh xạ sang SampleError::DeviceBusy (sample_error) và rơi vào nhánh DeviceBusy|Capacity trước nhánh tôi thêm. opt49: ghi dòng bên trong nhánh DeviceBusy khi out.is_stopped() + xả ngay dòng cancelled. Quay về: opt47.

## opt49 triển khai 12:10; QUYẾT ĐỊNH 12:20: QUAY VỀ 1 WORKER GPU (opt50, tự triển khai sau round kế)
- Người dùng: opt47 mất 11 phút để ra -5084 trong khi đối thủ thắng ở ngưỡng -5084 chỉ sau 4–7 phút. Phân tích: (1) nonce ≤-5084 của ta ~0,5–0,6/giờ (12 job sâu/giờ x ~5%), đối thủ ~6 khối/giờ ⇒ chênh ~10x, ~3–5x sau khi chia phần cứng 5090/4090 (~1,4x) ⇒ hiệu suất thuật toán/thông lượng của họ vượt xa bản của ta (suy luận từ số liệu, không biết nguyên nhân). (2) Hai worker làm mỗi job sâu 52→~104 s; mỗi lần round bị hủy mất gấp đôi công việc dở dang (≈7%→≈14% GPU trong round 6 phút); lợi ích lấp khoảng rảnh KHÔNG thấy trong số đo (GPU 77%, p10 48 so với opt44 72–82%). ⇒ hai worker không có lợi chứng minh được, có chi phí rõ ⇒ mặc định QS_GPUW=1 (opt50 = opt49 + QS_GPUW mặc định 1). Quay về: opt49 (2 worker) hoặc opt44.
- Ý tưởng thay thế (chưa làm): tách vai — 1 worker deep độc quyền (ưu tiên stream cao), worker sàng chạy tầng A/B/C với ưu tiên thấp, chuyển salt qua kênh deep ⇒ giữ độ trễ deep 52 s và lấp khoảng rảnh bằng việc sàng (ngắn, hủy rẻ). Cần tách screened_reads thành sàng + deep.

## ĐO CỠ QUẦN THỂ SAU THU NHỎ (12:25–12:33 VN 15/11; miner dừng 8 phút, bật lại với opt50) — KẾT QUẢ ÂM, ĐỪNG LẶP LẠI
- Harness instance 0, deep 2048 reads x 131072 sweeps, shrink ở 35% về N read, 2 seed (401, 402): N=256: -5691,00 / 47,1 s; N=384: -5691,48 / 54,3 s; N=512: -5689,68 / 55,0 s; N=768: -5685,82 / 63,2 s.
- Kết luận: giữ nhiều read hơn sau thu nhỏ KHÔNG cải thiện năng lượng (n=2, sai số ~±2) và thời gian tăng gần tuyến tính (+15% cho 384, +17% cho 512, +34% cho 768) ⇒ pha sau thu nhỏ đã kín GPU (256 block lấp đầy SM), không có SM rảnh để dùng miễn phí. Giữ 256.
- Giữ nguyên số liệu: script scratch/shrinkto.sh, shrinkto.tsv.
- opt50 (1 worker, mặc định QS_GPUW=1; ghi nhật ký hủy/round) chạy từ 12:33 (miner bật lại sau đo; symlink đã đặt trước).
- Hồi cứu chain 119 (người dùng nhắc 11 lane msa): QUIP_MSA_MAX_WIDTH=11 = số nonce msa chạy song song (1 SM/nonce, 11/128 SM); phần còn lại cho quip-screen (~14 triệu nonce/s, stage0/1/Q); bài học 'nút thắt là SCREEN' (NOTES.md của repo quip-miner-cuda dòng 34). Chain 120: sàng đầu S0 CPU ~1800 inst/s + tầng A ~17 salt/s ⇒ thông lượng sàng thấp hơn ~1000x. Hai hướng: (1) deep SM-giới hạn (block lặp qua nhiều read) + worker sàng ưu tiên thấp, ước +10–15%; (2) sinh instance + sàng stage0 bit-parallel trên GPU (cổng bộ sinh twisted_dimer ~3750 dòng; trần tương quan lý thuyết ~0,86 từ phương sai instance 25 vs seed 9, hiện 0,3).

## KẾ HOẠCH LỚN: SINH INSTANCE + SÀNG S0 TRÊN GPU (người dùng duyệt 12:45 VN 15/11: 'nếu bước tiến lớn thì nên thử')
- Đo CPU từng phần mỗi salt (test gen_timing, 300 nonce, 1 luồng, máy đang chạy miner): generate_instance 11,1 ms; features S0 2,55 ms; ising_graph 0,70 ms ⇒ ~14,4 ms ⇒ 32 luồng ≈ 2200/s (thực tế ~1800/s).
- Vì sao đáng làm (ước lượng, chưa đo): độ lệch chất lượng giữa instance σ_I ≈ 4,3 (live); S0 ở q≈1/90 (z≈2,3) chọn thêm ≈ z·ρ·σ_I = 2,4 đơn vị với ρ=0,24; sinh nhanh 100x cho phép S0 ở q≈1/9000 (z≈3,7): ≈ 3,8 đơn vị ⇒ +1,4 ở đầu vào, ~+1 ở kết quả deep (sau khi tầng A/B/C chồng lấp) ⇒ tần suất đuôi x2,5 (mỗi đơn vị ≈ x2,5). Cùng bậc ứng viên mô phỏng tốt nhất (x3, sim_lease). Công thức chain 119: thông lượng sàng thô lớn + chọn gắt.
- Cấu trúc bộ sinh (ising-generators 0.0.15, ice/twisted_dimer.rs, ice/build.rs, rng.rs): 5 luồng ChaCha8 (seed = BLAKE3 derive_key 'quip family stream v1' trên nonce+descriptor digest+topology digest+nhãn; mỗi nonce 5 seed, CPU tính được vài µs); left/right shuffle (Fisher–Yates, uniform bằng loại bỏ) ⇒ select tile; gauge 4 bit/tile; dấu yếu theo thứ tự cạnh không-tile; pair shuffle ~28,5k phần tử rồi duyệt tham lam 1200 cặp. ~80k từ ChaCha mỗi instance (~5k khối); mẫu phần mềm tuần tự trên mỗi instance nhưng song song giữa các instance ⇒ ước 1–2e5 instance/s trên 4090 (băng thông hoán đổi ~3,6 MB/instance, L2 72 MB).
- AN TOÀN: GPU chỉ dùng để CHẤM ĐIỂM S0; nonce qua lọc được dựng lại bằng generate_instance chuẩn trên CPU ⇒ sai lệch GPU chỉ làm lọc nhiễu, không làm nonce sai.
- Mốc: M1 bản CPU 'lean' khớp bit với generate_instance (1000 nonce) [cũng cho ~3x CPU]; M2 kernel CUDA khớp bit với M1; M3 đặc trưng S0 trên GPU; M4 tích hợp (worker GPU sinh+chấm điểm, danh sách nonce qua S0 → luồng CPU dựng đầy đủ → funnel hiện tại); M5 đo trên miner thật. Mỗi mốc ghi vào đây.

## M1 XONG (12:40 VN 15/11): BỘ SINH GỌN TRÊN CPU KHỚP BIT VỚI generate_instance
- src/cluster/lean_gen.rs (LeanGen): cache candidates/sides một lần; mỗi nonce: 5 luồng ChaCha8 (seed BLAKE3 stream_seed của crate), 2 shuffle left/right, select (Kuhn của crate), gauge, dấu yếu, shuffle cặp dimer ⇒ vector j theo thứ tự cạnh đã sắp. Test lean_matches_canonical (ignored, long-tests): 300/300 nonce j GIỐNG HỆT, edges giống; 2,05 ms so với 12,03 ms (x5,9) khi miner đang chạy.
- PHÁT HIỆN: tham số THẬT của họ chain 120 (family-c27a60af1f03.json): zephyr 12x4, weak=45, pairs=1350 (recipe x-w45-p1350) — KHÔNG phải 62/1200 như harness (CU4_SHAPE=12,4,62,1200). Kết quả harness vẫn dùng được cho so sánh tương đối nhưng không phản ánh đúng bài toán.
- Phần còn tốn: features S0 2,55 ms (cần tối ưu: danh sách tam giác tính sẵn), ising_graph 0,7 ms, select của crate (BTreeMap + Kuhn). Bước kế: (a) tích hợp vào luồng sinh (lean → S0 → chỉ nonce qua S0 mới gọi generate_instance chuẩn), (b) tối ưu features, (c) cổng CUDA (M2).

## opt51 (12:46 VN 15/11): CHẤM ĐIỂM S0 NHANH — tự triển khai sau round kế (deploy51.sh)
- Bảng tam giác tính sẵn (33.729 tam giác; Triangles::features) cho đặc trưng S0: GIỐNG HỆT features_raw, 0,13 ms so với 3,24 ms (x25). Bộ sinh gọn + đặc trưng: ~2,5 ms/salt so với ~17 ms (≈x6,8 thông lượng sàng S0).
- lease_run.rs: lean_for(family) dựng LeanGen+Triangles một lần cho mỗi descriptor, tự kiểm tra 4 nonce với generate chuẩn (lệch ⇒ tắt, QS_LEAN=0 cũng tắt); luồng sinh chấm điểm S0 từ j gọn; CHỈ salt qua S0 (hoặc thăm dò 1/20000) mới gọi generate chuẩn + dựng đồ thị. Bộ điều khiển S0 theo hàng đợi sẽ tự siết q để GPU vẫn ăn đủ (kỳ vọng q ~1/90 → ~1/600).
- Kỳ vọng (ước lượng): z từ ~2,3 lên ~3,0 ⇒ +0,7·ρ·σ_I ≈ +0,7 đơn vị ở đầu vào (ρ=0,24, σ_I=4,3) ⇒ ~+0,5 deep ⇒ tần suất đuôi x1,4–1,6. Mốc đánh giá: pass-rate S0 (log), salts/phút, mức dùng GPU, số job sâu/10 phút, trung vị/đuôi. Quay về: opt50.

## PHÂN RÃ TỪNG HẠNG MỤC (13:00–13:20 VN 15/11; Nsight Compute + bộ đếm PROF; người dùng yêu cầu 'kiểm tra từng phần, tối ưu từng hạng mục')
- Thời gian mỗi lần chạy (harness, 12x4 w45 p1350, miner dừng; PROF_RUN): A 256x64: init 2,5 ms, kernel 7,9 ms, PA host 0,3 ms (post chỉ đáng kể khi QC_KEEP=0: 14 ms; sản xuất KEEP=1 ⇒ ~0,5 ms ⇒ A ≈ 11 ms thật, KHÔNG 23 ms); B 256x8192: kernel 788 ms + PA host 49 ms (5,8%); C 512x16384: kernel 3214 + PA 126 (3,8%); deep 2048x32768: kernel 11502 + PA 275 (2,3%).
- Phân bổ thời gian luồng GPU/phút (tỷ lệ qua A→B 3%, B→C 6%, C→deep 17%; 1070 salt/phút ở A): A ≈15%, B ≈47%, C ≈11%, deep ≈27% (tổng 46–62 s/phút khớp log). Phần ngoài kernel (init+PA host+post) chỉ ≈10% ⇒ tối ưu host tối đa vài %. B là hạng mục lớn nhất.
- ncu (quip_cluster_anneal, 256 block x 512 thread, SM 2,23 GHz base): IPC 1,89/4, issue slot 49,6% bận, 0,82 warp eligible/scheduler (7,8 active); occupancy lý thuyết 66,7% (thanh ghi 48 ⇒ 2 block; shared 43 KB ⇒ 2 block); stall: long scoreboard 3,62, barrier 3,27, wait 2,73, short scoreboard 1,73, branch resolving 1,64; shared: 32% wavefront thừa (4 lệnh LDS gather s[v]/parent[v] chiếm ~46% phần thừa; trung bình ~3,4 wavefront/yêu cầu), ATOMS (CAS union-find) chiếm ~11% phần thừa; global: 16% sector thừa (LDG.128/64 của ELL); branch efficiency 82,6%; một lệnh BRA 1 lần/luồng chiếm 15% mẫu stall (chờ barrier đầu kernel).
- THỬ 1 — sắp xếp lại thứ tự nút trong lớp màu để tránh xung đột bank (layout_opt.rs, test layout_study): mô hình chi phí 2,343x ideal → 2,069x (300k bước, T≈0 hoặc 0,3; 600k bước nóng chỉ 2,31x) ≈ −20% phần thừa; ncu ước tính tối đa +31% nếu xóa HẾT phần thừa ⇒ thực tế ≲6%; CHƯA tích hợp (ưu tiên thấp).
- THỬ 2 — nạp 4 hàng xóm liên tiếp trước khi dùng (vòng metropolis và vòng trường yếu; kernel cluster.cu, kết quả nguyên số): thời gian A -0,5%, B -1,2%, C -1,5% ⇒ KHÔNG ĐÁNG (độ trễ L2 đã được che bởi 24 warp/SM). ĐÃ HOÀN TÁC kernel (bản gốc giữ trong scratch/cluster.cu.opt51).
- Kết luận: kernel đã sát giới hạn kiến trúc (stall phân tán: scoreboard, barrier, union-find atomic); vi tối ưu cho ≤ vài %. Đòn bẩy còn lại ở cấp phễu/thuật toán: (a) sinh+lọc S0 nhanh (opt51 CPU x~7; cổng GPU), (b) thay đổi cấu hình tầng B (47% thời gian): ngắn hơn/ít read hơn/ q khác — cần mô phỏng với dữ liệu thật.
- Công cụ để lại: QC_PROF=1 (in PROF_RUN mỗi lần chạy), QC_LINEINFO=1 (NVRTC --generate-line-info cho ncu), pa_host_ms/kernel_wait_ms trong log từng phút của miner (cần triển khai build mới).

## opt51 ĐO CƠ CHẾ (13:30 VN 15/11; người dùng hỏi 'cần mẫu opt51 để làm gì')
- Thông lượng sàng thực (tổng salts_done của 'lease done' theo 15 phút): trước opt51 ~190–330 salt/s (09:30–12:30), sau opt51 (12:48+): 791 → 1859 → 1330 salt/s (x5–7). (Con số '1800 inst/s' cũ trong ghi chú là khả năng CPU thô, KHÔNG phải thông lượng thực qua LEASE_GATE/lease.)
- Mức siết S0 trong screen_state.json: s0_q_milli=85463 ⇒ q≈85 (cân bằng q ≈ G/tiêu thụ GPU ≈ 1500/18). Trước opt51 q≈17 (≈ q0 mặc định 16, G≈300/s). ⇒ S0 chặt hơn x5: z từ ~1,9 lên ~2,5 ⇒ ước +0,6·ρ·σ_I ≈ +0,6 đơn vị đầu vào ≈ +0,4 deep (ρ=0,24, σ_I=4,3). Sửa ước lượng trước ('q 1/90 → 1/600' sai vì lấy nhầm mốc 1800/s làm thông lượng cũ).
- Phát hiện sai số mẫu: để phân biệt hiệu ứng +0,4 đơn vị với sd 3,25 cần ≈ (2·3,25/0,4)² ≈ 260 attempt/nhánh ≈ 24 giờ ⇒ KHÔNG thể đánh giá opt51 bằng chất lượng nonce trong vài giờ. Dùng chỉ số cơ chế (q, salt/s, GPU, lỗi) để quyết định; giữ opt51.
- Đòn bẩy kế: q cân bằng tỉ lệ thuận thông lượng sinh+chấm điểm G ⇒ cổng GPU (nhanh x10–100) đẩy q lên ~850–8500: z 2,5 → 3,1–3,7 ⇒ thêm +0,4–0,8 đơn vị đầu vào.

## M2 (GPU) + PHÁT HIỆN NÚT THẮT THẬT CỦA SINH INSTANCE (13:40 VN 15/11) — opt52 TRIỂN KHAI 13:41
- kernels/gen_s0.cu + src/cluster/gen_gpu.rs: bộ sinh + chấm điểm S0 trên GPU (một block/instance; ChaCha8, shuffle Fisher–Yates, ghép cặp tối đa Kuhn, gauge, dấu yếu, cặp dimer, đếm tam giác). KHỚP: 256/256 nonce điểm S0 trùng khít (chênh tối đa 0). Tốc độ lần đầu 7,5k nonce/s (≈34 ms/instance/block, 256 block): chu kỳ/instance (nghìn): Kuhn 17.200, cặp dimer 15.750, dấu yếu 8.200, gauge 3.900, hai shuffle đầu 1.800, chấm điểm 285. Chưa tối ưu (đường để lại: nạp adjacency vào shared đè lên mảng pair, tạo j của shuffle song song bằng chunk + lane 0 áp hoán đổi, gauge/weak song song bằng truy cập ngẫu nhiên của ChaCha). CHƯA tích hợp vào miner (không cần nếu CPU đủ).
- NÚT THẮT THẬT: 24 luồng sinh chạy 30–100% CPU nhưng chỉ ra ~1,5k salt/s. Đo test fast_flags_matches_and_scales (miner đang chạy): đường sinh gọn CÓ cấp phát (couplings + select của crate: vector lớn + BTreeMap) với 24 luồng = 1.745 salt/s (đúng bằng thông lượng thật), trong khi cùng mã 1 luồng 2,2 ms/salt ⇒ tranh chấp cấp phát/mmap giữa các luồng. Đường KHÔNG CẤP PHÁT (LeanGen::fast_flags: Kuhn trên CSR, bộ đệm dùng lại, flags thay j; Triangles::features_flags): 0,66 ms/salt/luồng, 8 luồng 9.423/s, 24 luồng 13.431 salt/s (x7,7), đặc trưng GIỐNG HỆT bản chuẩn 300/300.
- opt52 = opt51 + đường không cấp phát (thread_local FastScratch) + tự kiểm tra 4 nonce (điểm lệch ⇒ tắt). Kỳ vọng: q cân bằng tăng từ ~85 lên ~500–700 (G tối đa ~13k/s, tiêu thụ GPU ~18/s) ⇒ z 2,5 → ~3,05 ⇒ +0,5·0,24·4,3 ≈ +0,5 đơn vị đầu vào (~+0,35 deep) không tốn GPU. Chỉ số kiểm: s0_q_milli trong /data/calib/screen_state.json, salt/s từ 'lease done', CPU luồng sinh, GPU. Quay về: opt51.

## THIẾT KẾ LẠI PHỄU CHO BỘ SINH NHANH (13:50 VN 15/11; người dùng: 'không dùng cách lọc của con kiến cho con voi')
- opt52 sau 3 phút: s0_q 85 → 512 (screen_state.json), CPU container 69% rảnh, GPU 85%, 411 W, không lỗi.
- scratch/sim_funnel3.py (copy ceiling/): mô hình mu~N(m0,4,3²) theo instance, mỗi tầng đo = tương quan cố định với kết quả deep (S0 0,24; A 0,39; B 0,61; C 0,70, chia 0,85 để ra tương quan với mu) + nhiễu độc lập; deep = mu + N(0, 2,7²); tầng lọc theo thứ hạng (cửa sổ cuốn); mục tiêu số nonce deep ≤ -5084,5 mỗi giờ GPU; chi phí A 12 ms, B 0,88 s, C 3,35 s, deep 47 s (+ follow-up khi ≤ -5081). Hiệu chuẩn m0=-5066: cấu hình cũ q0=17 (A32,B16,C4) cho trung vị deep -5080,2 (khớp thực -5080). Thực tế deep/h ~12, mô hình 22 (bỏ hao phí hủy round) ⇒ chỉ dùng so sánh TƯƠNG ĐỐI; đuôi mô hình hơi nặng (3,7 sự kiện/h so với thực ~0,5).
- Kết quả (cấu hình cũ 32,16,4): q0 17 → 85 → 512 → 1000: sự kiện/h 3,74 → 4,92 → 6,34 (+70%) → ~6,8; trung vị deep -5080,3 → -5080,8 → -5081,75. Độ nhạy theo ρ_S0: 0,15: +43% (q0 17→512), 0,24: +70%, 0,35: +90%.
- Tinh chỉnh lại q tầng (qA,qB,qC) ở q0≥512: tốt nhất (1000; 32,16,2) 7,03 so với cấu hình cũ ở q0=512 6,50 ⇒ chỉ +8% (trong sai số mô hình) ⇒ KHÔNG đổi q các tầng (không đáng một lần restart). Đòn bẩy chính còn lại là chất lượng bộ dự báo S0 (ρ), không phải tầng sau.
- Hướng kế: nâng ρ của S0 bằng đặc trưng giàu hơn (CPU còn 69% rảnh và 13k salt/s dư địa): ví dụ mật độ vòng 4-cạnh, tương quan dấu yếu quanh tile, hoặc 1–2 sweep SA CPU rất rẻ trên đồ thị đã nén; mỗi +0,1 ρ_S0 ≈ +20% sự kiện/h theo mô hình.

## opt52 sau 22 phút và opt53 (14:05 VN 15/11)
- opt52 (22 phút): s0_q kẹt đúng 512,000 = TRẦN trong mã (clamp 512_000 milli) trong khi CPU 93% rảnh ⇒ bộ điều khiển chạm trần, chưa cân bằng; thông lượng 5.748 salt/s (10 phút); GPU 82% (p10 48); 9 attempt thật (4,1/10 phút so với 1,8 của opt44), tốt nhất -5080,605, trung vị -5078,35 (mẫu nhỏ). Nghi vấn: cửa sổ xếp hạng tầng C (256 mẫu, ~2 mẫu/phút) còn nặng mẫu của chế độ cũ (instance kém hơn) ⇒ ngưỡng C lỏng hơn ⇒ nhiều job sâu hơn (4,1/10 phút) nhưng mỗi job kém chọn lọc; sẽ tự điều chỉnh sau ~2 giờ khi cửa sổ cuốn hết.
- opt53 = opt52 + trần q của S0 512 → 4096 và cửa sổ điểm S0 2048 → 8192 (phân giải phân vị 1/4096 bằng 2 mẫu; chi phí quét 8192 phần tử/salt dưới mutex ≈ 10% một lõi ở 13k salt/s). Mô phỏng: q0 512→1000 ≈ +7%. Triển khai 14:04 khi nonce round -5078,9 (target ~-5087). Quay về: opt52.

## opt54 TRIỂN KHAI 14:50 VN 15/11: PHỄU TỐI ƯU CHO ĐUÔI (mục tiêu mỗi round có nonce ≤ -5085; nonce yếu hơn vô dụng)
- Người dùng nhấn mạnh: kết quả round là nonce sâu nhất / tối thiểu -5085; đối thủ -5085…-5088; mục tiêu 'mỗi round' (không phải mỗi giờ). Với xác suất mỗi round nhỏ, P≈λ·t_round nên thứ hạng cấu hình theo 'sự kiện ≤ T mỗi giờ GPU' giữ nguyên (cộng hao phí hủy round).
- sim_funnel4.py (ceiling/): mô phỏng trực tiếp từng dòng dõi (promise, kmax), mục tiêu T=-5085: ở q0=512 cấu hình cũ (A32,B16,C4, promise -5081, kmax 2) = 6,34; tốt nhất (A64,B16,C2, promise -5082, kmax 4) = 7,66 (+21%); ở q0=1000: cũ 6,51 → tốt nhất 7,78 (+19%). A64 có mặt ở cả 5 cấu hình đầu; kmax 4 và promise -5082/-5081 phổ biến. (Mô hình chỉ so sánh tương đối.)
- opt54 = opt53 + mặc định QS_STAGES 256:64:64;256:8192:16;512:16384:2, QS_PROMISE -5082, QS_FOLLOW 4 (tối đa 4 dòng dõi phụ). Triển khai khi nonce round -5082,6 (target ~-5087). Quay về: opt53 (QS_STAGES cũ 256:64:32;...:4, promise -5081, follow 2).
- Theo dõi: ở opt52/53 (55 phút, 19 attempt) KHÔNG có nonce ≤ -5083 (kỳ vọng ~3,4 theo opt44; xác suất ~3%): chưa kết luận; nếu sau ~40 attempt nữa vẫn không có ≤ -5083 thì quay về opt44 và kiểm giả thuyết 'S0 chọn lệch' (Goodhart: S0 là mô hình tuyến tính huấn luyện trên tầng B ngắn; ở đuôi cực đoan có thể chọn instance chỉ tốt ở sàng nông).

## opt55 (15:02) và PHÉP ĐO NHÃN 'ĐÁY' CHO S0 (15:08 VN 15/11)
- 28 attempt của opt52–54 (13:41–15:00): 0 nonce ≤ -5083 (xác suất ~0,4–3% nếu như opt44), trung vị -5078,3 so với -5079,6, dù các salt đến tầng C tốt hơn ~1,3 (p50 -5062,9 → -5064,2). Nghi vấn S0 chặt chọn 'instance dễ ủ nhanh' chứ không phải 'đáy sâu' (S0 học từ năng lượng tầng B ngắn). Trong 199 job deep của calib, tương quan S0–deep chỉ 0,09 NHƯNG bị co hẹp bởi chọn lọc tầng sau (không kết luận được).
- opt55 (hoàn tác an toàn): q tối đa của S0 = QS_S0MAX mặc định 20 (trước 4096), phễu về (A32,B16,C4, promise -5081, follow 2). Giữ đường sinh nhanh không cấp phát. Triển khai 15:02 khi nonce round -5081,0 (target -5086,9). Quay về opt54/opt53 nếu cần.
- Thử nghiệm đo trực tiếp (GPU, miner dừng, trần 1h30): harness CU4_FAMILY (topology + tham số họ thật 4560 nút), CU4_NONCE_FILE (1000 nonce ngẫu nhiên), CU4_SCAN chế độ quét in 'SCAN idx năng_lượng nonce đặc_trưng'. Giai đoạn 1: 900 instance, 512 read x 16384 sweep (kiểu tầng C, ~4 s/instance), PA32. Mục tiêu: tương quan giữa điểm S0 và đặc trưng với năng lượng đáy trên instance KHÔNG bị chọn lọc; sau đó giai đoạn 2 trên tập con với deep sâu hơn để phân biệt 'dễ ủ nhanh' và 'đáy sâu', rồi huấn luyện lại S0 theo nhãn đáy. Giám sát tự động labelwait.sh (dừng nếu đứng >10 phút). Script: scratch/labelrun.sh.

## KẾT QUẢ NHÃN ĐÁY + ĐUA GIẢI THUẬT + DASHBOARD ĐỐI THỦ (15:00–16:05 VN 15/11)
- Dashboard công khai đối thủ (API /api/telemetry, /api/mining/attempts/<n> của tu17nodeslove): ~93 attempt/s, năng lượng attempt nông -4984,6 ± 5,0 (480/500), ~4% attempt sâu hơn (-5015…-5035, vài cái -5068), mỗi attempt báo qpu_access_time ~25 s (thời gian lô, nhiều nghìn nonce song song), kết quả round tốt nhất -5084,7 từ ~47.700 attempt ⇒ phễu nhiều tầng: nông (~-4985) → ~-5023 → ~-5068 → ~-5085. Ước ~1,8 giây-SM/nonce ở tầng nông (170 SM / 93 mỗi giây). Dữ liệu BudFox lẫn bản ghi chain 119 cũ (cùng số solution, thang -14.xxx) nên KHÔNG dùng được.
- Phép đua (5 instance chain 120 thật; msa 1 SM/nonce; Otsuka cả 128 SM): msa 128x4096: -4949 @1,48 giây-SM; msa 128x16384: -4975 @5,95; msa 128x65536: -5002 @23,8; Otsuka 256x256: -4951 @5,9; 256x1024: -5005 @15,8; 256x4096: -5032 @55; tầng A hiện tại 256x64: -4834 @~1,4. ⇒ ở -4950 msa rẻ hơn Otsuka ~4x; ở -5000 ngang nhau. Đối thủ ~1,8 giây-SM cho -4985 vs msa ~9 ⇒ họ hơn msa chưa tinh chỉnh ~5x (~3x sau khi trừ 5090/4090). (Sửa: ước '10–25x' trước đó SAI vì ngoại suy từ tầng A kém.)
- Nhãn đáy (437 instance ngẫu nhiên, 512x16384): S0 tương quan 0,268 (Pearson; 0,254 Spearman), 10% S0 thấp nhất tốt hơn 10% cao nhất 3,78 đơn vị; huấn luyện lại 9 đặc trưng: 0,196 (tệ hơn S0 ⇒ không có thêm thông tin trong 9 đặc trưng). msa 128x1024 (0,37 giây-SM): 0,308; msa 128x4096: 0,296; tầng A Otsuka (1,4 giây-SM): 0,220; kết hợp S0+msa4096: 0,334; S0+msa4096+A: 0,354. msa1024 vs msa4096 chỉ 0,28 với nhau (nhiễu seed lớn, trần do nhãn nhiễu).
- sim_msa.py: thay tầng A bằng msa (ρ~0,55 suy từ tỉ lệ 0,308/0,220 x 0,39; 2,9 ms toàn GPU) ở q0=20: phễu tối ưu (A128,B8,C8) 8,09 so với 4,38 của phễu tối ưu hiện tại (+85%); ρ bi quan 0,45: +68%. Giả định chưa kiểm: (1) ρ của msa với deep (mới đo với nhãn kiểu C), (2) chạy 128 nonce song song hiệu quả như bench 1 nonce, (3) msa là kernel persistent chiếm hết SM (cần chia SM như chain 119 'width 11').
- Khả thi tích hợp: streaming::run_stream(device, kernel, jobs: Receiver<StreamJob>, out: Sender<StreamResult>, cancel, gov, stall) đã có cho đường mining của quip-cuda-msa (slot song song = stream_width = SM/SM-mỗi-nonce). Cần: luồng msa riêng (CudaDevice::open_with_nodes(0, Msa, 4600)), giới hạn max_sms để chia SM với Otsuka, đẩy instance qua kênh, nhận năng lượng, hàng đợi top 1/qA sang B/C/deep.
- Việc kế: (a) đo ρ(msa) với deep thật trên ~100 instance (msa 128x1024 + Otsuka deep rút gọn; ~25 phút GPU), (b) thử nghiệm run_stream trên instance chain 120 thật: thông lượng nonce/s với 128 và 64 SM, (c) nếu đạt thì tích hợp lease_run (stage A = msa).

## TƯƠNG QUAN THẬT VỚI DEEP + opt56 (TẦNG ĐẦU msa) — 16:05–16:50 VN 15/11
- Deep rút gọn (2048 read x 32768 sweep, thu nhỏ 35%→256) trên 100 instance ngẫu nhiên (tb -5066,1; sd 3,33). Tương quan Pearson với deep: S0 0,408; tầng A Otsuka (256x64) 0,100; msa 128x1024 0,211; msa 128x4096 0,332; tầng B kiểu (256x8192) 0,498; tầng C kiểu (512x16384) 0,439 (sai số ~±0,09). ⇒ A gần như vô dụng; C KHÔNG hơn B dù đắt gấp 4; B là tầng Otsuka đáng giá nhất.
- S0 BÃO HÒA (437 instance): năng lượng đáy theo phân vị S0: top5% -5058,8; 5-10% -5059,1; 10-20% -5058,9; 20-40% -5058,2; 40-60% -5057,0; 60-80% -5057,0; 80-100% -5055,8 ⇒ chọn gắt hơn ~top 40% không thêm gì ⇒ giải thích opt52–54 không hơn (q 340–512 vô ích); mô phỏng 'q0 17→512 +70%' SAI (giả định ρ không bão hòa). Giả thuyết 'S0 chọn lệch' (Goodhart) KHÔNG đúng: S0 dự báo deep tốt (0,41) nhưng chỉ ở nửa trên.
- Thông lượng msa run_stream (1000 instance thật khác nhau, 128 nonce trong tầm bay): 128x1024: 111,5 nonce/s (-4895); 128x4096: 85,7 nonce/s (-4942). Chi phí xử lý mỗi job ~9 ms giới hạn 1024 sweep.
- sim_funnel5.py/sim_msa2.py (ρ đo được; S0 bão hòa q0=3): phễu hiện tại 0,67 sự kiện(≤-5085)/h (khớp thực ~0,9/h); phễu Otsuka tối ưu (A16,B32,C2, promise -5079, kmax 4) +25%; thay A bằng msa 128x4096 (ρ 0,33, 11,7 ms): tốt nhất (A256,B16, bỏ C, promise -5079, kmax 4) 1,91 so với 0,80 của A tốt nhất ⇒ +140%; msa 1024 +116%; ρ bi quan 0,25: +90%.
- opt56 (build 16:44, triển khai 16:46): lease_run.rs thêm msa_mode()/msa_dev() (CudaDevice::open_with_nodes(0, Msa, 4600) + UtilGovernor), msa_worker: lô QS_MSAK=1024 salt qua S0 stream vào run_stream (128 read x 4096 sweep, QS_MSAR/QS_MSAS), xếp hạng theo năng lượng msa, giữ 1/QS_MSAQ=256 tốt nhất (≥1) sang tầng Otsuka (generator không dựng PreparedGraph, bỏ Instance; sống sót dựng lại từ nonce), phần còn lại out.screen. Mặc định khi msa: QS_STAGES=256:8192:16 (B rồi deep, bỏ A và C), QS_PRIOR=-5052.5;-5064, promise -5079, follow 4; trạng thái sàng riêng /data/calib/screen_state_msa.json. QS_MSA=0 tắt (quay về opt55). Quay về: ln -sfn /data/quip-cuda-otsuka-opt55 + docker restart.
- Chỉ số theo dõi opt56: số phiên msa/phút (mong ~5/phút), attempt sâu/10 phút, năng lượng attempt, lỗi/panic, GPU, S0 q.

## SỬA NHẬT KÝ + opt57 (17:00–17:05 VN 15/11)
- Lỗi nhật ký (người dùng báo: khối 485 cách đây 8 phút, 486 cách đây 14 phút): (1) script cancelled_to_attempts.py ghi vào thư mục N theo log trong khi coordinator ghi round 'qblock_id=N' của log vào thư mục N-1 ⇒ lệch +1 so với dòng thật của coordinator; (2) coi cả lần dừng container (docker stop trong các phép đo GPU, restart khi triển khai) là round bị hủy. Sửa: thư mục = qb_log−1; chỉ nhận hủy nằm trong 30 s sau 'new round' (chủ round = round ngay trước); fix_sidecar_rows.py chuyển 7 dòng về đúng khối, loại 39 dòng do dừng container. Script chạy lại 17:00.
- Nonce -5085,915 (16:30:23, round 482): ngưỡng lúc đó -5086,205 (người đi trước), thua 0,29; trùng lúc dừng miner cho phép đo (không phải nguyên nhân). Ngưỡng thực tế quanh -5086…-5087 (trung bình ~-5085,9) ⇒ mục tiêu thiết kế là ≤ -5086/-5087, KHÔNG phải -5085.
- sim_tail.py (ρ đo được; round 420 s; P = 1−exp(−sự kiện/h × 420/3600)): P(có nonce ≤T trong 1 round) với T=-5085/-5086/-5087: phễu cũ 7,6/4,7/3,0%; Otsuka tối ưu 10,8/6,4/4,1%; msa tối ưu 21,5/14,7/12,3% (A512,B8,C2, promise -5081, follow 6 cho T sâu); cấu hình opt56 (A256,B16,bỏ C,promise -5079,follow 4) 17,0/11,4/7,4%.
- opt57 = opt56 + msa chọn theo cửa sổ xếp hạng cuốn 4096 năng lượng msa (mặc định giữ 1/512, QS_MSAQ) thay vì top theo lô; tầng Otsuka B (256x8192 q8) + C (512x16384 q2); promise -5081, follow 6. Triển khai 17:03 khi nonce round -5079,7 (target ~-5086). Quay về: opt56 / opt55.
- opt56 16 phút (16:46–17:02): 7 attempt, tốt nhất -5079,665, trung vị -5077,55 (mẫu nhỏ).

## NHẬT KÝ ROUND ĐƯA VÀO COORDINATOR (17:10 VN 15/11; người dùng: 'sao không đưa vào coordinator cho dễ quản lý')
- Lý do trước đây dùng script phụ trên host: tránh đụng coordinator đã kiểm chứng (đổi phải restart container), và năng lượng dở dang của miner chỉ miner biết. Hạn chế đã thấy: script phải đoán khối từ log nên lệch +1 khối so với quy ước coordinator và nhầm cả lần dừng container.
- coordinator 0.3.9-rc5-patched3 (cây scratch/wt-rc5; bản vá coordinator-rc5-patched3.patch): CoordinatorState thêm round_best_milli/round_salts/round_started_ms, cộng dồn ở handle_lease_done (chỉ lease thuộc round hiện tại, tốt nhất = min best_energy_milli của LeaseDone, số salt đã sàng); stop_mining (đổi round) ghi MỘT dòng AttemptRecord::round_summary cho round vừa kết thúc: qblock_id = st.qblock_id (đúng quy ước các dòng khác của coordinator), raw_best_energy = năng lượng sàng tốt nhất của round, accepted/submitted=false (hiện 'rejected' trên dashboard), device_access_time_us = độ dài round (µs), job_id 64 hex bắt đầu bằng 32 số 0. Không cần script phụ nữa: đã dừng cancelled_to_attempts.py (giữ bản sao ở ceiling/). Triển khai 17:11 khi nonce round -5076,4 (target ~-5087). Quay về: ln -sfn /data/quip-coordinator-0.3.9-rc5-patched2 + docker restart quip-cuda.
- Chưa có trong dòng tóm tắt: số salt (Attempts) — AttemptRecord không có trường đếm; round_salts có sẵn trong CoordinatorState nếu sau này muốn thêm vào summary/attempts.json.

## msa Ở NGÂN SÁCH SÂU (17:11–17:20 VN 15/11; miner dừng 8 phút) — msa KHÔNG thay được Otsuka ở mức sâu
- msa nonce song song (128 instance chain 120 thật, 128 read): 262144 sweep: -5019,2 (sd 3,6; min -5027,7), 86 giây/nonce trên 1 SM (1,5 nonce/s toàn GPU), tương quan với deep Otsuka rút gọn 0,544; 1048576 sweep: -5033,8 (sd 3,9; min -5046,6), 343 giây-SM (0,37 nonce/s), tương quan 0,539.
- So cùng chi phí (giây-SM): Otsuka B 256x8192 ≈113 giây-SM: -5047; C 512x16384 ≈420: -5057; deep rút gọn 2048x32768 ≈1470: -5066; msa 1M sweep 343: -5034 ⇒ ở ≥100 giây-SM Otsuka hiệu quả hơn msa ~5–6x (256x4096 Otsuka -5032 @55 giây-SM vs msa -5034 @343). Giao điểm hiệu quả ~10–20 giây-SM (msa 65536: -5002 @24; Otsuka 256x1024: -5005 @16).
- Kết luận: tầng 1 (rẻ, <~10 giây-SM) dùng msa (opt56/57 đúng hướng); các tầng từ ~100 giây-SM trở lên giữ Otsuka. Tầng 2 bằng msa 262144 (ρ 0,54 @86 giây-SM) chỉ ngang tầng B Otsuka (ρ 0,50 @113) ⇒ không đáng đổi. Đối thủ tu17: tầng 2 ≈-5023 ≈ msa 262144 (-5019), nhưng tầng 3 (-5068) và cuối (-5085) vượt xa mọi thứ msa làm được ⇒ họ có bộ giải mạnh ở tầng trên (chưa biết), không chỉ msa.

## LỊCH NHIỆT ĐỘ msa — PHÁT HIỆN LỚN + opt58 (17:20–17:45 VN 15/11)
- Quét beta của msa 128x4096 trên 128 instance thật (nonce song song): mặc định -4942,7; (0,05;2) -4851; (0,1;3) -4935; (0,3;8) -4960; (0,5;8) -4964; (1,0;8) -4971; (1,2;8) -4972,7; (2,0;20) -4933; quét mịn: (1,5;5,0) -4974,4 (tốt nhất), vùng bằng phẳng lo 1,2–1,5, hi 4–8 ≈ -4972…-4974. Cùng chi phí (86 nonce/s).
- Tương quan với deep Otsuka rút gọn (n=100): msa (1,5;5,0) 0,500; (1,2;8) 0,484; mặc định 0,332 ⇒ ngang tầng B Otsuka (0,50) nhưng ~75 lần rẻ hơn (1,5 so với 113 giây-SM).
- sim_tail2.py (T=-5085, round 420 s): opt55 7,6%; opt57 (msa ρ 0,33, A512 B8 C2) 18,6%; msa chỉnh beta ρ 0,50: 32,0% (A256 B4 C2 promise -5081 follow 4, deep/h 31); ρ bi quan 0,42: 27,3% (A512 B8 C2 promise -5079 follow 6).
- opt58 = opt57 + beta_range (1,5;5,0) cho job msa (QS_MSAB), QS_MSAQ 256, tầng Otsuka B (256x8192) q4 + C (512x16384) q2, promise -5081, follow 4. Triển khai 17:43 khi nonce round -5080,6 (target ~-5085,9). Quay về: opt57.
- Bài học: lịch nhiệt độ mặc định của sampler cách xa tối ưu cho họ chain 120 (30 đơn vị ở msa). Việc kế: quét beta/sweeps_per_beta cho Otsuka (tầng B, deep), có thể còn chỗ tương tự.
- opt57 (khoảng 25 phút chạy thực): 10 attempt, tốt nhất -5082,655, trung vị -5078,52.

## opt59 (17:52 VN 15/11): cấu hình phễu bền vững cho mục tiêu SÂU (T -5086…-5087)
- sim_cmp.py (round 420 s; ρ_msa 0,50 / bi quan 0,42): T=-5086: opt58 (A256 B4 C2 f4) 20,3% / 13,7%; A512 B8 C2 -5081 f6 20,5% / 16,5%; T=-5087: opt58 11,4% / 8,4%; A512 B8 C2 -5081 f6 14,3% / 12,6%. Các cấu hình A512 B8 C4 -5079 f6, A1024 B8 C2, A512 B4 C2 kém hơn. Với T=-5085 lạc quan (ρ 0,50) opt58 32% vs ~ngang. ⇒ chọn A512, B 1/8, C 1/2, promise -5081, follow 6.
- opt59 = opt58 với QS_MSAQ 512, QS_STAGES 256:8192:8;512:16384:2, QS_FOLLOW 6 (msa beta (1,5;5,0) giữ). Triển khai 17:52 khi nonce round -5080,7 (target ~-5086). Quay về: opt58 / opt57.

## MỤC TIÊU BÁM THEO ĐỐI THỦ (17:55 VN 15/11; người dùng: 'cập nhật target đối thủ thắng, họ luôn tiến bộ')
- API công khai /api/blocks (100 khối gần nhất, 04:19–17:52 VN): năng lượng thắng trung vị theo giờ: 04h -5083,65; 05h -5084,36; 06h -5084,78; 07h -5085,18; 08h -5086,49; 09h -5084,69; 10h -5084,84; 11h -5085,49; 12h -5084,35; 13h -5085,71; 14h -5086,94; 15h -5085,09; 16h -5086,25; 17h -5086,86. 3 giờ gần nhất (22 khối): trung vị -5086,25; p25 -5087,24; p75 -5085,20; sâu nhất -5088,06; độ dốc hồi quy 12 giờ -0,12/giờ; dự báo +6 giờ -5086,96. Người thắng 3 giờ: BudFox 11 (tb -5086,73; thời gian đến khi thắng tb 460 s), tu17 10 (tb -5085,86; 244 s), 1 khác. /api/miner-wins: BudFox 190 thắng (avgMiningTime 82 s), tu17 171 (231 s), người thứ ba 81 (best -5086,545).
- Công cụ: scratch/winning_targets.py (copy ceiling/) → targets.json; dùng mỗi lần kiểm tra. Mục tiêu thiết kế chuyển từ -5085 sang trung vị -5086,25 (chính) và p25 -5087,25 (sâu), kèm dự báo.
- sim_cmp.py với T=-5086,25 / -5087,25 (ρ_msa 0,50 / 0,42), P(có nonce ≤T trong 1 round 420 s): opt59 (A512 B8 C2 -5081 f6) 17,7% / 16,0% và 13,3% / 10,2%; opt58 17,9% / 12,5% và 10,3% / 7,9% ⇒ giữ opt59.

## Mục tiêu thiết kế = trung vị thắng − (1…2) (18:20 VN 15/11; người dùng yêu cầu biên an toàn)
- Nguồn: API cục bộ localhost:20049/api/blocks?limit=500. 3h: trung vị −5086,40; thiết kế nhẹ −5087,4, chính −5087,9, sâu −5088,4.
- P(≥1 nonce ≤T / round 420 s), ρ_msa 0,50 / 0,42: T −5087,4: opt58 9,7/7,2%; opt59 (A512 B8 C2 −5081 f6) 12,5/10,2%. T −5087,9: opt58 6,7/5,3; opt59 10,0/8,8. T −5088,4: opt58 4,7/4,4; opt59 8,3/7,3. Biến thể khác (C4, A1024, B4) đều kém opt59 ⇒ giữ opt59. Còn dư địa: lịch ủ Otsuka (beta/sweeps) là hướng kế.

## LỊCH SỬ THẮNG THEO PHIÊN BẢN (18:25 VN 15/11; người dùng nhớ: 8h–14h thắng xen kẽ -5082…-5084)
- API cục bộ: ta thắng 08h 3 khối (-5085,6…-5086,5; opt44), 11h 1, 12h 1, 13h 1 (-5084,5…-5085,2; opt45–opt51); từ 13:41 (opt52) đến 18:25 (opt52→59, ~5 giờ): 0 khối, trong khi đối thủ chỉ nhích ~1 đơn vị. Kỳ vọng ~1 khối/giờ ⇒ P(0 trong 5h) ≈ 0,7%. Bằng chứng trùng khớp ghi nhớ 'mọi thay đổi từ sau opt51 đều giảm khối thắng' (opt52–54 S0 q lớn, opt55 lean/GPU gen, opt56–59 msa).
- Trung vị attempt không phân biệt các bản (−5078…−5080) ⇒ lợi thế nằm ở đuôi sâu mà trung vị không thấy.
- Quyết định: quay về opt51 (bản cuối có khối thắng, rollback51.sh tự chạy sau khi quét beta kết thúc). opt56–59 giữ làm nhánh nghiên cứu.
- Quét beta Otsuka tầng B (256x8192, 100 instance): 1,2-8 spb32 trung vị -5047,06; spb8 -5046,15; spb128 -5046,23 (không khác biệt).

## opt60 (18:30 VN 15/11; người dùng: KHÔNG quay về opt51, nghi hướng sau opt51 sai ở đâu đó làm lệch msa) — ĐO TƯƠNG QUAN msa→deep TRÊN DỮ LIỆU THẬT
- Đã hủy rollback51. Phát hiện: nhật ký hiệu chuẩn live KHÔNG ghi năng lượng msa; chỉ có ~54 dòng chế độ msa trong 3500 dòng cuối ⇒ chưa từng xác nhận trên dữ liệu thật rằng msa chọn đúng (ρ 0,50 chỉ đo trên harness).
- opt60 = opt59 + (1) dòng calib thêm "msa" (năng lượng msa) và "mx" (cờ lấy mẫu ngẫu nhiên); (2) QS_MSAX (mặc định 2048): 1/2048 salt bị msa loại vẫn đi qua phễu (B, C, deep nếu đạt) để đo tương quan msa↔B/C KHÔNG thiên lệch chọn lọc (~150 mẫu/giờ ⇒ SE ρ≈0,08). Chi phí ~+0,05 job B/s.
- Cách đọc: calib.jsonl dòng có "mx":1: tương quan (msa, st[0]=B) và phân vị; so với harness ρ(msa,deep)=0,50 / ρ(B,deep)=0,50.
- Quét beta Otsuka tầng B (256x8192, 100 instance thật): trung vị 1,2-8 spb32 -5047,06; spb8 -5046,15; spb128 -5046,23; 0,8-8 -5044,17; 2,0-8 -5027,50; 1,2-5 -5047,28 ⇒ mặc định đã gần tối ưu (khác msa), không còn +30 ở đây.
- Quét beta B XONG (18:31), 10 cấu hình, trung vị: 1,2-8 spb32 -5047,06 (mặc định, tốt nhất); spb8 -5046,15; spb128 -5046,23; 1,2-5 -5047,28; 1,2-12 -5045,84; 0,8-8 -5044,17; 2,0-8 -5027,50; 2,5-12 -5005,14; (1,5-10 và 0,5-6 xem bscan/). KẾT LUẬN ÂM: lịch ủ Otsuka mặc định đã tối ưu, đừng quét lại.
- opt60 triển khai 18:31:59 (đo tương quan msa↔B không thiên lệch, QS_MSAX 2048).
- Thống kê theo phiên bản (attempt cuda-0): ≤-5083 mỗi giờ: opt44–51 1,9/h (17 trong 8,9h); opt52–59 0,22/h (1 trong 4,5h, p≈0,2%); trung vị không đổi (-5078…-5080) ⇒ đuôi sụt từ opt52. Giả thuyết: S0 q (opt52–54 q 512–4096; opt55 cap 20 vs opt51 q≈85) rồi msa. Thử tiếp: S0MAX≈85 sau khi opt60 đủ mẫu.

## opt60 ĐO msa↔B TRÊN DỮ LIỆU THẬT (19:03 VN 15/11; 30 phút chạy, mẫu ngẫu nhiên n=29)
- corr(msa,B)=0,32 (SE≈0,17; msa sd 4,4; B sd 4,0). Survivor (n=35): msa TB -4993,3, B TB -5053,5; mẫu ngẫu nhiên: msa TB -4975,5, B TB -5047,6 ⇒ msa tốt hơn 18 đơn vị đem lại B tốt hơn 6,0 (hệ số 0,33). KẾT LUẬN SƠ BỘ: msa CHỌN ĐÚNG trên dữ liệu thật, không có lỗi lệch rõ; chưa đủ mẫu để phân biệt ρ 0,3 và 0,5. Cần ≥100 mẫu.
- Thắng 19:02: 0 khối từ 18:00 (ngưỡng -5085…-5088); attempt opt60: 9, tốt nhất -5080,715.

## opt61 TRIỂN KHAI 19:13 VN 15/11 (người dùng: 'chờ mẫu làm gì, trung vị vẫn lẹt đẹt -5076…-5080') — THỬ TRỰC TIẾP GIẢ THUYẾT 'S0 q / msa làm sụt đuôi'
- opt61 = opt60 code với mặc định QS_MSA tắt (QS_MSA=1 mới bật) và QS_S0MAX 85 ⇒ phễu như opt44–51 (A32,B16,C4, promise -5081, follow 2) nhưng bộ sinh nhanh. Chỉ tiêu: ≤-5083 ≥ ~1,5/giờ (opt44–51: 1,9/h; opt52–59: 0,22/h). Đánh giá sau ~3 giờ. Quay về msa: ln -sfn /data/quip-cuda-otsuka-opt60.

## ĐỐI CHIẾU HARNESS ↔ MINER THẬT + BỘ DỰ BÁO PHI TUYẾN (19:25–19:50 VN 15/11; miner dừng 2 phút)
- 100 nonce live (calib.jsonl, tầng B 256x8192) chạy lại trong harness: live TB -5049,61 (sd 4,82) vs harness -5049,94 (sd 4,30); lệch TB -0,33 (KHÔNG thiên lệch); tương quan từng nonce 0,37; sd chênh 5,13 ⇒ nhiễu seed lớn: tầng B chỉ lặp lại được 0,37 (là giới hạn của bộ lọc, không phải lỗi harness). Công cụ: scratch/livecmp.sh, nonces_live100.txt, live100_ref.json.
- Dự báo phi tuyến từ 9 đặc trưng loại tam giác (calib_features, 6000 nonce live, nhãn B): corr(S0 live, B)=0,155; ridge CV -0,007; GBR CV 0,107; +phổ 0,092 ⇒ 9 đặc trưng KHÔNG chứa thêm tín hiệu ngoài S0 (trong vùng đã lọc S0). Phân vị S0 top5% so với dưới: B tốt hơn chỉ 2,2. KẾT LUẬN: S0 đã bão hòa; sinh nhanh hơn không giúp vì không có bộ dự báo rẻ nào mạnh hơn; sức chọn nằm ở bộ phán xét giải-thử (A, msa) tốn GPU. Giải thích 'sinh nhanh nhưng trung vị không đổi'. Hướng thực sự còn lại: đặc trưng MỚI (không phải 9 lớp tam giác) hoặc tăng ρ của phán xét rẻ.

## opt61: THẮNG KHỐI ĐẦU TIÊN SAU 6 GIỜ (19:32:48 VN 15/11, nonce -5087,265, sâu hơn p25 thắng -5087,24) — 20 phút sau triển khai (S0 q 85, msa tắt). Mẫu 1; tiếp tục theo dõi tần suất ≤-5083.

## opt61 theo dõi 20:02 VN 15/11 (0,8 giờ từ 19:13)
- Thắng: 1 (19:32, -5087,265); thua #506 (-5087,7; ta -5080,3), #507 (-5085,4; ta -5083,56), #508 (-5085,5; ta -5085,015 — thua 0,5). Attempt 19 dòng (có dòng trùng nonce), nonce khác nhau ≤-5083: 3 (-5087,27; -5085,02; -5083,56) ⇒ ~3–4/giờ (opt44–51: 1,9/h; opt52–59: 0,2/h). Mẫu nhỏ nhưng ủng hộ giả thuyết msa/S0 q làm sụt đuôi. Giữ opt61.

## opt62 TRIỂN KHAI 20:12 VN 15/11: TẦNG TRUNG GIAN D (đề xuất từ phễu 4 tầng của tu17; người dùng: 'ok hãy tiến hành'; lưu ý: đối thủ thắng sau attempt 4–12 phút)
- sim_tierD.py (T=-5086,4 / -5087,4; P/round 420 s): opt61 (A32 B16 C4 f2) 0,34/h (3,9%) / 0,20/h (2,3%); thêm tầng D 512x32768 (ρ 0,55–0,65, 6,7 s): +47…+72% (0,50–0,54/h; 0,30–0,34/h); D 2048x32768 (ρ 0,77): +72% / +103%. Cấu hình tốt: B16 C2 D4–8, promise -5081…-5079, follow 4. Tuyệt đối vẫn thấp (4–7%/round) so với đối thủ (thắng sau 4–12 phút ≈ P/round cao hơn nhiều).
- opt62 = opt61 + QS_STAGES "256:64:32;256:8192:16;512:16384:2;512:32768:4", QS_PRIOR thêm -5068, QS_FOLLOW 4 (mặc định chế độ không msa). Không đổi mã nào khác. Quay về: ln -sfn /data/quip-cuda-otsuka-opt61. Chỉ tiêu: ≤-5083 ≥ opt61 (~3/h) và số job sâu giảm (~6–12/h); đánh giá sau ≥2 giờ.

## HƯỚNG 6 'PROMISE THÍCH ỨNG THEO NGƯỠNG ROUND' (20:30 VN 15/11) — KẾT QUẢ MÔ PHỎNG: KHÔNG CÓ LỢI, KHÔNG TRIỂN KHAI (ĐỪNG LẶP LẠI)
- opt63 (đã build, KHÔNG deploy; /data/quip-cuda-otsuka-opt63): promise = target + QS_PROMISE_REL (mặc định 5), kẹp [-5084,-5077]; QS_PROMISE tuyệt đối vẫn ghi đè.
- sim_adapt.py (phễu opt62, ngưỡng T lấy mẫu -5084,5…-5088 theo phân bố thắng): P/round trung bình: promise cố định -5081 4,60%; rel +5 4,49%; rel +4 3,54%; rel +6 4,57% ⇒ khác biệt trong nhiễu mô phỏng. P/round phụ thuộc chủ yếu vào ngưỡng round: ~10% ở T=-5084,5, ~5,5% ở -5085,5…-5086, ~3% ở -5086,5, ~2,5% ở -5087…-5088 (không điều chỉnh được).
- Kết luận: promise cố định -5081 đã gần tối ưu; thích ứng theo ngưỡng không thêm. Giữ opt62 chạy tiếp (đánh giá 22:10).

## Theo dõi 20:33 VN 15/11: THẮNG LẦN 2 TRONG 1H (20:33:23, nonce -5086,855, opt62 sau 22 phút). opt61 58 phút: 16 attempt, 5 sâu, ≤-5083: 3, ≤-5085: 2 (-5087,27; -5085,02), 1 khối thắng; opt62 22 phút: 6 attempt, 3 sâu, ≤-5083: 1, thắng 1. Tổng từ 19:13 (80 phút): 2 thắng, 4 nonce ≤-5083 khác nhau (~3/h). Giữ opt62, đánh giá 22:10.

## opt62 theo dõi 21:02 VN 15/11 (51 phút từ 20:11): 10 attempt (nonce khác nhau), 6 sâu, tốt nhất -5088,185 (SÂU NHẤT TOÀN CHAIN 3 giờ qua), ≤-5083: 3, ≤-5085: 3 (-5088,19; -5086,86; -5085,42); THẮNG 2 khối (20:33 -5086,855; ~20:5x -5088,185); thua #514 sát (ta -5085,415 vs -5086,145). Từ 19:13 (1h50): 3 thắng. Giữ opt62; đánh giá chính thức 22:10.

## TRUNG VỊ NONCE TỐT NHẤT MỖI ROUND THEO PHIÊN BẢN (21:20 VN 15/11; người dùng: 'xem Recent Performance, opt61/62 trung vị tốt hơn')
- (attempts/<q>: min raw_best mỗi round; chỉ round có bản ghi) opt35-40: -5081,44 (21 round); opt41-43: -5081,41 (26); opt44: -5080,84 (34; ≤-5085 12%); opt45-51: -5080,87 (16); opt52-55: -5080,24 (15; ≤-5083 7%); opt56-59 (msa): -5079,66 (15; tệ nhất, ≤-5085 0%); opt61: -5080,88 (7; ≤-5085 29%); opt62: -5082,72 (7; p25 -5086,85; ≤-5083 43%; ≤-5085 43%). ⇒ người dùng đúng: trung vị theo round của opt61/62 tốt nhất (mẫu nhỏ 14 round); msa làm TỆ hơn (-5079,7).

## RÚT NGẮN DEEP (sim_deeplen.py, 21:35 VN 15/11) — KHÔNG LỢI: deep 131072 (47 s) P/round 4,2% (T -5086,4); 65536 (24 s, +0,8 shallower) 2,5–3,1%; 32768 (12 s, +2,0) 1,0%. T -5087,4: 2,3% vs 0,9–3,2% vs 1,2%. Giữ 131072.

## ĐO TẦNG C/D/E TRÊN 60 NONCE NHÃN (21:22–21:48 VN 15/11; miner dừng 26 phút) — E KHÔNG ĐÁNG
- Tương quan với deep rút gọn (n=60, SE≈0,1): C 512x16384 (3,7 s) 0,55 TB -5057,6; D 512x32768 (7,5 s) 0,68 TB -5062,8; E 512x65536 (14,7 s) 0,67 TB -5067,2. E gấp đôi chi phí D mà không tăng tương quan ⇒ KHÔNG thêm tầng E. Công cụ: scratch/tierE.sh, tiere/.
- sim_cd.py với ρ đo thật (A 0,10; B 0,50; C 0,55@3,7 s; D 0,68@7,5 s): T -5086,4: cấu hình hiện tại opt62 (B16 C2 D4, promise -5081, follow 4) 0,28/h (3,2%/round); tốt nhất B16 C4 D3–6, promise -5079, follow 6: 0,58–0,65/h (6,6–7,3%) ≈ +100%. T -5087,4: hiện tại 0,25/h; B32 C3 D8 -5081 f6 0,42/h hoặc B16 C4 D3 -5079 f6 0,38/h (+50–70%). Nhiễu sim ±20% nên chỉ tin xu hướng: C chặt hơn (q4), D q3–6, promise -5079, follow 6.
- Ứng viên opt64 (CHƯA triển khai; chờ opt62 có ≥2 giờ dữ liệu, tránh restart): QS_STAGES 256:64:32;256:8192:16;512:16384:4;512:32768:4, QS_PROMISE -5079, QS_FOLLOW 6.

## opt64 TRIỂN KHAI 21:58 VN 15/11 (người dùng: 'trung vị đã tăng; tỷ lệ thắng là may; quay lại bản nào cũng được; cần bứt phá nhanh hơn')
- opt64 = opt62 + C q2→4 (512:16384:4), QS_PROMISE -5079 (cố định; bỏ promise thích ứng của opt63), QS_FOLLOW 6. Tầng: A 256x64 q32, B 256x8192 q16, C 512x16384 q4, D 512x32768 q4, deep 2048x131072. Cơ sở: sim_cd.py với ρ đo thật (+100% ở T -5086,4: 3,2% → 6,6–7,3%/round). Chỉ tiêu (3 giờ): ≥ -5081,5 trung vị nonce tốt nhất mỗi round, ≥25% round có ≤-5085, ≤-5083 ≥3/h. Quay về: ln -sfn /data/quip-cuda-otsuka-opt62 (hoặc opt61).

## XEM LẠI TỔNG THIẾT KẾ PHỄU + opt65 "msa LẶP NHIỀU LẦN" (22:00–22:40 VN 15/11; người dùng: cần bứt phá trung vị -5085…-5088)
- Dữ liệu thật (calib.jsonl, 229 job sâu phễu cũ): B -5058,7; C -5068,5; deep -5079,1 (sd 2,9). Hồi quy deep ~ 0,28·B + 0,24·C, sd dư 2,75 ≈ nhiễu seed ⇒ trong số job sâu, gần như toàn bộ biến thiên là nhiễu seed; chất lượng instance của entrant ≈ -5079 (≈2,2σ trên trung bình -5069,5). Oracle trên cùng lưu lượng (4,5k ứng viên/round → 2 deep) sẽ được ≈ -5083,7 ⇒ mất ~4,6 đơn vị do bộ phán xét nhiễu. Mỗi +1 đơn vị chất lượng entrant ≈ ×2,2 xác suất một lượt deep đạt -5086,5 (s=2,9: -5079 0,5%; -5081 2,9%; -5083 11%).
- Vì sao giai đoạn msa (opt56–59) kém: mẫu KHÔNG lệch (10% dòng) cho B của survivor msa (1/512) = -5050,3 ≈ survivor S0+A (-5050,6) (ngẫu nhiên -5047,6) ⇒ một lần msa chỉ ngang tầng A; trong khi opt59 nới B q16→8, C q4→2 ⇒ entrant deep kém hơn (B -5057,2; C -5067,2; deep -5077,8 n=16, đúng bằng dự báo từ B,C). Lỗi THIẾT KẾ (nới tầng sau), không phải lỗi msa. (Con số 'survivor B -5053,5' trước đó bị lệch chọn lọc.)
- Ma trận tương quan (437/100/60 nonce nhãn): mô hình MỘT nhân tố khớp; tải lên deep: A 0,10; msa 1 lần 0,38; B 0,50; C 0,44–0,55; D 0,68; E 0,67. Độ lặp lại: C↔C 0,51; B live↔harness 0,37; msa 2 seed 0,36.
- ĐO MỚI msa lặp (msarep.sh; 437 nonce x 8 seed, beta 1,5–5): trung bình k lần msa 128x4096 ↔ deep: k=1 0,38; 2 0,41; 4 0,49; 8 0,54 (↔C n=437: 0,37/0,50/0,54/0,57; ↔D 0,50→0,66). Lấy MIN không tăng (0,36) ⇒ phải dùng TRUNG BÌNH. msa 64x4096 x8: 0,62 (n=100), 115 nonce/s. ⇒ TB 8 lần msa (≈0,1 s GPU) ≥ tầng B (0,88 s) và C (3,7 s): độ chính xác trên mỗi giây GPU cao hơn ~8–35 lần. R(deep ~ msa8+B)=0,61.
- pop×seed (popseed.sh, deep 65536 sweep, 8 instance): 512→64: -5068,6 (15 s); 1024→128: -5070,4 (29 s); 2048→256: -5069,9 (26 s, 1 seed); sd seed 1,9–2,1. Cùng chi phí không lợi ⇒ KHÔNG đổi quần thể (xác nhận ghi chú cũ 'đảo không giúp').
- sim_new.py (tham số đo thật; mô hình lạc quan ~2,7 đơn vị so với thực tế nhưng dùng để so tương đối): phễu hiện tại entrant deep -5081,8, P/round 22%; phễu msa lặp (x1 giữ 1/8 → +3 giữ 1/8 → +12 giữ 1/4 → B q4 → D q4 → deep): -5083,0…-5083,6 (+1,2…+1,8), P/round 52–53% (×2,9–3,0).
- opt65 TRIỂN KHAI 22:37: msa_worker nhiều lượt (QS_MSAP1 8, QS_MSAK2 3, QS_MSAP2 8, QS_MSAK3 12, QS_MSAP3 4; xếp hạng theo TRUNG BÌNH trong lô 1024; dựng lại đồ thị survivor song song 8 luồng; msa_session()), QS_MSA mặc định bật, QS_STAGES msa "256:8192:4;512:32768:4", QS_PRIOR "-5054.5;-5069", promise -5079, follow 6, S0MAX 85, trạng thái /data/calib/screen_state_msa2.json. QS_MSAREP=0 về luật 1 lần. Quay về: ln -sfn /data/quip-cuda-otsuka-opt64 (hoặc opt62).
- 3 phút đầu opt65: 12 phiên msa, GPU 97%, không lỗi; job sâu đầu tiên: msa TB -4985,2, B -5058,7, D -5072,7, deep -5078,9.
- Việc kế: (1) theo dõi opt65 ≥2 giờ (trung vị nonce tốt nhất/round, tỉ lệ ≤-5085); (2) điểm kết hợp (msa TB + B + D) thay cho xếp hạng từng tầng; (3) thử msa 64 read (rẻ hơn 35%, chất lượng tương đương); (4) gộp 3 lượt vào một phiên stream (bớt ~2,4 s/lô).

## PHÂN TÍCH tu17 LẦN 2 (22:45–23:03 VN 15/11; API công khai /api/mining/attempts/<n> chỉ trả 500 attempt cuối; lấy mẫu định kỳ 10 cửa sổ = 73 s, script rival/poll.sh)
- Tầng 1: ~69–90 nonce/s, -4985 ± 5, qpu ~25 s (≈2000 nonce chạy song song). Tầng 2: 3,9/s (≈5%), -5017…-5030, qpu 25–50 s, theo lô 10–14. Tầng 3: 0,23/s ≈ 100 kết quả/round, năng lượng TB -5075,0, sd 5,1 (‑5066…‑5087), qpu TB 210 s (≈50 nonce tầng 3 chạy đồng thời). Round 529 họ thắng sau 8,2k attempt; nonce thắng -5086,3 nằm trong một lô 9 nonce tầng 3 (8 cái -5072…-5076).
- Mô hình thắng của họ: ~100 lượt giải vừa (-5075, sd 4,5–5) mỗi round ⇒ P(≤-5086,5) ≈ 0,26%/lượt × 100 ≈ 23%/round (khớp nhịp thắng ~3 khối/giờ). Của ta: 2–4 lượt deep/round (-5079,5, sd 2,9) ⇒ 0,8%/lượt ⇒ 2–5%/round.
- Với bộ giải Otsuka của ta, deep 47 s vẫn tốt hơn E 15 s trên mỗi giây GPU (0,017%/s so với 0,005%/s) ⇒ KHÔNG sao chép 'nhiều lượt vừa' bằng Otsuka. Nhưng chi phí mỗi lượt tầng 3 của họ ước ≈ 2–3,5 s GPU 5090 cho -5075 (ta: E 15 s) ⇒ bộ giải tầng sâu của họ hiệu quả hơn ~3–4 lần và chạy song song theo nonce (giống msa ủ dài).
- GIẢ THUYẾT MỚI (chưa từng thử): msa ủ DÀI với beta đã chỉnh. Phép đo 'msa ở ngân sách sâu' (17:11) dùng beta MẶC ĐỊNH (-5019 @262144, -5034 @1M) — trước khi phát hiện beta (1,5;5) cho +30 ở 4096 sweep. Đang đo msalong.sh: 128x65536/262144 với beta (1,5;5), (1,2;8), (2;6) và 128x1M (1,5;5) trên 128 nonce nhãn.
- opt65 sau 25 phút: 6 job sâu: -5078,9; -5083,5; -5078,0; -5076,4; -5081,4; -5079,3 (TB -5079,6 ≈ phễu cũ -5079,1; n nhỏ); entrant B -5056,3, D -5072,9; 80 phiên msa/25 phút.

## msa Ủ DÀI VỚI BETA ĐÃ CHỈNH (23:03–23:16 VN 15/11; miner dừng 13 phút) — KẾT QUẢ ÂM, ĐỪNG LẶP LẠI
- 128 nonce nhãn, năng lượng TB: 65536 sweep (22 giây-SM): beta (1,5;5) -5014,5; (1,2;8) -5015,5; (2;6) -5011,2. 262144 sweep (86 giây-SM): -5026,0; -5029,9; -5028,2. 1M sweep (342 giây-SM, 1,5;5): -5035,7. Chỉnh beta chỉ +10 ở ngân sách sâu (so mặc định -5019 @262144; -5034 @1M).
- So với Otsuka cùng chi phí: B 256x8192 (113 giây-SM) -5047; msa 262144 (86) -5030 ⇒ Otsuka tốt hơn ~17. msa ủ dài KHÔNG thay được tầng Otsuka (giả thuyết 'tầng sâu song song theo nonce kiểu tu17 bằng msa' bị bác). Cấu hình tầng sâu của tu17 vẫn chưa giải thích được.
- Danh sách hướng còn lại (23:12): opt65 theo dõi; điểm kết hợp msa+B+D; tầng cuối rẻ/song song theo nonce (chưa có cách); bộ giải cuối mới (SB/directed-loop); tối ưu kernel Otsuka (3 khối/SM, +10–30%); tinh chỉnh opt65 (msa 64 read, gộp 3 lượt).

## KẾ HOẠCH ĐÊM 15→16/11 (23:20 VN; người dùng đi ngủ, trao toàn quyền như đêm trước; cron kiểm tra mỗi 30 phút)
- Quy tắc quyết định (tự động, không chờ người dùng): 
  1. opt65 chạy liên tục tới ~01:00 (≥25 round). Chỉ số: trung vị nonce tốt nhất mỗi round, tỉ lệ round có ≤-5085, số khối thắng. So với baseline opt62/64 (trung vị round -5082,7; ≤-5085 43% mẫu 7 round; opt61/62 cộng lại 2 thắng/9 round). Nếu trung vị round opt65 < -5080,5 VÀ tỉ lệ ≤-5085 < 15% sau ≥25 round ⇒ quay về opt64 (ln -sfn /data/quip-cuda-otsuka-opt64; docker restart quip-cuda). Ngược lại giữ.
  2. Không restart vì lý do khác trong ≤2 giờ sau mỗi lần đổi bản (cửa sổ sàng). Khi đang thắng liên tiếp (≥2 khối trong 1 giờ) không đổi gì.
  3. Việc không dùng GPU được làm bất cứ lúc nào (điểm kết hợp msa+B+D; msa 64 read; gộp 3 lượt) nhưng chỉ triển khai sau khi opt65 có kết luận; mỗi lần chỉ đổi một nhóm tham số.
  4. Chuỗi thua ≥3 khối + nonce tốt nhất round thấp hơn target >6: ưu tiên quay về opt64/opt62 (bản có khối thắng), không thử bản mới giữa đêm.
  5. Đo GPU (dừng miner) chỉ khi có hướng rõ và tổng ≤ 20 phút; luôn có trap bật lại; không chạy khi vừa thắng.
- Cập nhật mục tiêu bằng winning_targets.py mỗi lần kiểm tra; ghi mọi thử nghiệm (cả âm) vào đây và sao lưu.
- CẬP NHẬT 23:25 (người dùng sửa quy tắc): ngưỡng toàn quyền = chuỗi KHÔNG THẮNG ≥5 khối liên tiếp (thay cho 3). Khi chuỗi ≥5: được tự tối ưu/đổi thuật toán/đo đạc/chạy thử/restart không cần hỏi. Khi chuỗi <5: chỉ giám sát; việc không đụng miner (phân tích, build, mô phỏng) vẫn làm. Quy tắc 4 ở trên đọc là 'chuỗi thua ≥5'. Quy tắc 1 (đánh giá opt65 ~01:00, quay về opt64 nếu kém) vẫn là quyết định tự động đã duyệt.

## opt65 BỎ — QUAY VỀ opt62 (23:37 VN 15/11; người dùng: 'bỏ opt65, không nonce đủ sâu, trung vị kém'; chuỗi thua 9 khối)
- opt65 chạy 55 phút (22:37–23:32, trừ 13 phút dừng đo): 7 round, nonce tốt nhất/round -5076,4…-5083,3 (trung vị ≈ -5078,9; 0 round ≤-5085); 8 job sâu TB -5079,0 (≈ phễu cũ -5079,1; mô phỏng dự báo +1,2…+1,8). 0 thắng.
- Quay về opt62 (bản có dữ liệu: 71 phút, 9 round, trung vị round -5082,7, 2 thắng, 3 round ≤-5085) thay vì opt64 (chỉ chạy ~5 phút, chưa có dữ liệu). Lý do chọn: dữ liệu thật > mô phỏng. msa lặp KHÔNG được dùng nữa (mã vẫn còn trong lease_run.rs, QS_MSA=1 để bật).
- Bài học: mô phỏng (một nhân tố, nhiễu độc lập) lạc quan; phễu không msa (S0 q85 + A/B/C/D) cho kết quả tốt nhất; mọi thử nghiệm msa (opt56–59, opt65) đều không tốt hơn. Hướng msa đóng.

## ĐỔI KPI (00:05 VN 16/11; người dùng: đối thủ thắng -5086…-5089, ta vẫn xem trung vị và số nonce ≤ -5085)
- KPI chính từ giờ: tỉ lệ round có nonce tốt nhất ≤ năng lượng người thắng round đó (≈ -5086,5; 30 khối gần nhất: min -5088,51; p25 -5086,93; trung vị -5086,43; p75 -5086,15; ≤-5087: 7/30; ≤-5088: 3/30; ≤-5089: 0/30) + khoảng cách TB tới người thắng + số khối thắng. Trung vị attempt/trung vị round và '≤-5085' chỉ là chỉ số phụ. Chỉ tiêu thiết kế: phân phối kết quả tầng cuối phải đạt ≤ -5087 với xác suất ≥ 1%/lượt (hiện deep: ~0,8%; tầng D: ~0).

## ĐÁNH ĐỔI reads↔sweeps Ở NGÂN SÁCH ~15 s (23:57–00:15 VN 16/11; 24 nonce nhãn, không thu nhỏ; miner dừng 18 phút) — ÍT READ + ỦDÀI TỐT HƠN
- Cùng chi phí (~15 s/nonce): 2048x16384 TB -5060,9 (sd 4,1); 1024x32768 -5064,7 (3,8); 512x65536 (mốc E) -5067,2 (4,9; min -5076,7); 256x131072 -5068,0 (3,5; min -5074,8). Cùng nonce so với E: +6,3 / +2,4 / 0 / −0,8 (sâu hơn). Tương quan với deep: 0,57/0,52/(0,67)/0,49. ⇒ SWEEP quan trọng hơn READ rõ rệt ở ngân sách này, bão hòa quanh 256–512 read. Độ rộng (sd) không tăng ở ít read (3,5 vs 4,9) ⇒ giả thuyết 'phân phối rộng do ít read' KHÔNG được xác nhận.
- Hệ quả cần thử: deep hiện 2048 read (thu nhỏ 256 @0,35) x131072 sweep; thử 512x262144 (≈ cùng chi phí 47 s) — chỉ tiêu: P(≤-5087) mỗi lượt cao hơn deep. (ghi chú cũ 'opt9 262144 = opt8 131072' đo với quần thể lớn.)

## DEEP Ở NỬA NGÂN SÁCH: ÍT READ + ỦDÀI KHÔNG HƠN (00:16–00:41 VN 16/11; 18 nonce, miner dừng 25 phút) — KẾT QUẢ ÂM, ĐỪNG LẶP LẠI
- 2048x65536 thu nhỏ 0,35→256 (kiểu deep hiện tại, 26 s): TB -5069,78 sd 3,75 min -5076,3 | 512x131072 (29 s): -5068,60 (kém 1,18, sd chênh 3,07) | 256x262144 (30 s): -5069,32 (kém 0,47). P(≤-5078) 1,4% / 1,0% / 1,1% (xấp xỉ chuẩn). Tương quan với nhãn deep1: 0,73/0,66/0,74.
- Kết luận: thiết kế deep hiện tại (quần thể lớn thu nhỏ) đã ngang hoặc tốt hơn các biến thể ít read ủ dài ở cùng chi phí; ở ngân sách 15 s ít read thắng (256x131072 > 2048x16384 6,9) nhưng ở ngân sách deep thì thu nhỏ thắng. Trục reads/sweeps/cỡ quần thể ĐÃ CẠN. sd kết quả không đổi (3,5–4,0) ⇒ phân phối rộng của tu17 không đến từ trục này.
- Tổng kết các trục đã cạn: lịch ủ (beta, hình dạng), reads/sweeps/quần thể, tầng trung gian (E), msa (ngắn, lặp, dài), S0/đặc trưng (9 loại tam giác), promise/follow-up thích ứng. Còn lại: thông lượng kernel (+10–30%), điểm kết hợp các tầng, bộ giải thuật toán mới (chưa có ứng viên có bằng chứng).

## SO SÁNH VỚI BudFox & tu17 THEO TỪNG ROUND (01:35 VN 16/11; người dùng: 'cả hai chỉ chạy 1 GPU 5090; không thấy so sánh với BudFox')
- Nguồn: telemetry công khai recentMiningSubmissions (có attemptCount, qpuAccessTimeUs, energy tốt nhất nộp của mỗi round). Round 530–549 (20 round): BudFox best mỗi round trung vị -5079,3; p25 -5085,2; đạt ngưỡng 5/20 (25%); ≤-5085: 6/20. Ta (13 round có dữ liệu, chủ yếu opt65/opt62): trung vị -5079,2; p25 -5082,6; đạt ngưỡng 0/13 (round 549: -5086,07 vs -5086,455, thua 0,4).
- ⇒ TRUNG VỊ NGANG BudFox; chênh nằm ở ĐUÔI PHẢI: tỉ lệ round ≤-5085 ~30% (Bud) vs ~8–15% (ta; opt62 9 round: 33%). Chênh ~2–2,5× xác suất đuôi (≈ +1,3 đơn vị chất lượng đuôi, mỗi +1 ≈ ×2,2), KHÔNG phải 10× như ước lượng trước. Chuỗi 22 thua gồm ~1h40 dừng miner đo + opt65 + restart.
- BudFox: attemptCount/round trung vị ~1890 (136–5151), qpu TB ~44 s/attempt ⇒ ÍT attempt NẶNG; tu17: 3,7k–54k attempt/round, ~25 s ⇒ NHIỀU attempt NHẸ. Cả hai (một 5090) thắng ~25–40% round ⇒ hai kiến trúc khác nhau cho kết quả cỡ nhau; 'tầng 3 ~100 kết quả/round' là suy luận từ cửa sổ 500 attempt cuối (có thể là đợt bùng phát).
- Chỉ tiêu: đưa xác suất round ≤ ngưỡng thắng từ ~10% lên ≥25% = +1,3…1,5 đơn vị chất lượng đuôi.

## opt64 TRIỂN KHAI 01:45 VN 16/11 (người dùng: 'đợi 25 round không có ý nghĩa, đối thủ sẽ lên -5090; có thời gian thì tối ưu')
- Nguyên tắc mới: KHÔNG chờ N round để quyết định. Quyết định dựa trên số đo ngoại tuyến nhanh (harness/replay, vài phút) về cơ chế (chất lượng entrant, phân phối tầng cuối); vòng thật chỉ để xác nhận và theo dõi. Chuỗi thua >5 ⇒ được đổi bản tự do.
- opt64 (C q4, promise -5079, follow 6) dựa trên tương quan đo thật (C 0,55; D 0,68). Quay về: opt62.

## opt66 TRIỂN KHAI 01:50 VN 16/11: XẾP HẠNG BẰNG ĐIỂM KẾT HỢP z-score (B+C+D) THAY VÌ NĂNG LƯỢNG TỪNG TẦNG (người dùng: 'bạn làm được gì khi đang chờ?')
- Phân tích ngoại tuyến trên 229 job sâu thật của phễu cũ (calib.jsonl; không cần GPU): tương quan thô B-deep 0,25 / C-deep 0,22 (mẫu đã chọn lọc). Theo phân vị TRONG số job đã sâu: C top25% → deep ≤-5083 12,3% (7/57); B top25% → 10/57 = 17,5%; điểm kết hợp B+C top25% → 19,3% (11/57) so với 1,8% (25–50%) và 6,1% (50–100%); toàn mẫu 8,3% (19/229). Nghĩa là xếp hạng theo tổng z-score cho tỉ lệ đuôi ~1,6× so với chỉ C ở cùng mức giữ.
- Mã: sampler.rs comb_on() (QS_COMB=0 tắt), combined_score_milli() với QS_ZSTAT mặc định B(-5050,6;4,1;w1,15) C(-5064,2;3,7;w0,9) D(-5071,4;3,4;w1,5) (hệ số từ hồi quy deep~0,28·B+0,24·C và ρ_D 0,68); tầng ≥B xếp hạng theo điểm kết hợp (cửa sổ 256 điểm kết hợp, prior 0); trạng thái lưu ở /data/calib/screen_state_comb.json (khác định dạng cũ); QS_MSA mặc định TẮT lại. Cấu hình còn lại = opt64 (C q4, D q4, promise -5079, follow 6).
- Lưu ý: mẫu 229 job đã qua bộ lọc cũ (giới hạn phạm vi) ⇒ ước lượng lợi ích có thể thổi phồng; kiểm chứng thật qua phân phối entrant (B,C,D,deep trong calib) sau ~1 giờ. Quay về: opt64 (ln -sfn /data/quip-cuda-otsuka-opt64).

## KIỂM LẠI '3 KHỐI/SM' + B Ở 128/64 READ (02:00 VN 16/11; người dùng: bản thử 3 khối/SM cũ có thể sai vì thuật toán chưa tối ưu; miner dừng 5 phút)
- Số đo cũ (occ3.tsv, n=4048 VỪA 3 khối/SM; chain 120 n=4560 với parent uint16: 3×(7·4560+1 KB)=98,8 KB ≤ 100 KB cũng vừa): lb3 chậm B256 +5% (0,76 vs 0,72), C512 +4%, nhanh deep pop2048 −7% (3,75 vs 4,03). Kernel cluster không đổi từ opt37 ⇒ kết quả còn đúng (đặc tính phần cứng, không phụ thuộc phễu). Ước lượng ròng cho phễu hiện tại: deep pha pop2048 ≈ 81% chi phí deep (~35% GPU) −7% ⇒ −2%; B/C (~50% GPU) +4–5% ⇒ +2,3% ⇒ ≈ 0. Không tái đo.
- B ở ít read (100 nonce nhãn, 256 → 128 → 64 read x 8192 sweep): thời gian/nonce 0,99 → 0,87 → 0,86 s (CHỈ −12% dù nửa/ một phần tư số read ⇒ kernel bị giới hạn độ trễ dưới 256 read); năng lượng TB -5046,6 / -5043,6 / -5039,9; tương quan với deep 0,54 / 0,43 / 0,48. ⇒ đi xuống 128/64 read tuần tự không lợi (chất lượng −3…−7 cho −12% thời gian).
- sim_b128.py: nếu chạy ĐƯỢC 2 nonce B 128 read đồng thời (nonce-song-song, chi phí 0,5 s/nonce, ρ 0,43): T -5086,4: 6,8% → 7,3% (+7%); T -5087,4: 4,1% → 5,6% (+37%); tuần tự (0,87 s): 6,9% / 3,5% (không lợi). Cần hỗ trợ chạy hai job cluster đồng thời trên GPU (opt47 hai worker từng âm vì độ trễ/hủy) — chưa làm; lợi ích mô phỏng nhỏ ở T -5086,4.

## THẮNG #551 (01:33:06 VN 16/11, nonce -5086,9, opt64 sau 10 phút triển khai) — kết thúc chuỗi ~28 khối không thắng (lần thắng trước 20:59 hôm qua). Ngưỡng round -5086,735 (người thắng #551 là 0x5abae3 -5086,735? — dòng log: 'won by another' 18:27:48 UTC; xem lại: khối của ta nộp 18:33:06 UTC = round #552). opt66 triển khai 01:48 (restart thêm 02:03 do đo B 128/64 read): 4 phút dữ liệu, chưa đánh giá.

## CPU SONG SONG + sim_comb.py (02:15 VN 16/11; người dùng hỏi 'việc đang chờ có thể dùng CPU không?') — CPU còn rảnh (32 luồng, container ~2 nhân): chạy mô phỏng quét q trên 20 nhân
- Mô hình một nhân tố với ρ đo thật (A .10, B .52, C .55, D .68): xếp hạng ĐỘC LẬP vs KẾT HỢP cho cùng mức tối ưu (~16–20% / 12–17% mỗi round ở T -5086,5; sai số mô phỏng ±2–3%) ⇒ điểm kết hợp KHÔNG hơn xếp hạng độc lập trong mô hình Gaussian (phân tích 229 job '1,6×' bị giới hạn phạm vi). Cấu hình hiện tại (A32 B16 C4 D4) chỉ 11–12%; tốt nhất ~16–20% ở A64 B16 C3 D2–3 (T -5086,5) và A64 B16 C4 D6/ A32 B8 C6 D3 (T -5087,5): hướng chung: A chặt hơn (64), C q3, D q2–3, promise -5079, follow 6.
- opt67 (đã build, CHƯA triển khai vì ta vừa thắng 01:33 — chuỗi <5 chỉ giám sát): opt66 + QS_STAGES 256:64:64;256:8192:16;512:16384:3;512:32768:3. Triển khai khi chuỗi thua ≥5 hoặc khi opt66 kém. Quay về: opt66/opt64.

## QUAY VỀ opt64 (02:22 VN 16/11; người dùng: opt66 không có bằng chứng tốt, opt64 vừa thắng #551 lúc 01:33)
- opt66 (điểm kết hợp z-score) chạy 01:48–02:22 (có restart 02:03 do đo B 128/64 read): ~4 attempt, tốt nhất -5079,5; mô hình sim_comb cho thấy điểm kết hợp KHÔNG hơn xếp hạng độc lập; chưa có bằng chứng lợi. Bản thắng 01:33 (-5086,885) là opt64. Giữ opt64 làm bản chính; opt66/opt67 (A64 B16 C3 D3 + kết hợp) lưu trong data, chỉ thử khi chuỗi thua ≥5.

## KPI KHOẢNG CÁCH TỚI NGƯỜI THẮNG (02:35 VN 16/11; round 545–559, 15 round có dữ liệu của ta)
- khoảng cách = best của ta − năng lượng người thắng (dương = thua): trung vị +7,9; TB +10,7; ≤0: 1/15 (#552 thắng, -5086,89); sát: #549 +0,4 (-5086,07 vs -5086,45); nhóm +5…+9: 7 round (#545 +7,5; #547 +7,0; #550 +5,2; #554 +7,7; #555 +7,0; #556 +8,8; #557 +7,9); nhóm +13…+20: 5 round (#546, #548, #551, #553, #559) + #558 +30,0.
- Round ngay sau restart rất yếu: #558 -5055,9, #559 -5066,6 (restart 02:13) so với -5072…-5080 bình thường ⇒ mỗi lần restart mất ~2 round (cửa sổ sàng/prior trống ~15–20 phút). Tối qua+đêm nay ~15 lần restart ⇒ mất ~30 round (~5 giờ). QUY TẮC: gộp thay đổi, mỗi lần restart phải có lợi kỳ vọng > mất 2 round (~15–20%/1 round × 2 ≈ 4–7% xác suất thắng/lần).
- Dịch phân phối +1,3 đơn vị sẽ chuyển khoảng 1/15 round gần ngưỡng (#549) thành thắng ⇒ ~+7% xác suất; muốn ~30% cần dịch ≥ +6…+8 (cả nhóm +5…+9) ⇒ cần cải thiện bộ giải/phễu rất lớn, không phải tinh chỉnh q.

## ĐÍNH CHÍNH 02:50 VN 16/11 (người dùng: 'trung vị gần đây tệ quá, thua cả opt62')
- Nonce tốt nhất mỗi round theo bản (20:11→02:45; gán bản theo thời điểm giữa round): opt62 16 round trung vị -5079,6 (p25 -5082,7; ≤-5083: 4); opt64 10 round trung vị -5075,2 (p25 -5081,0; ≤-5083: 1); opt65 7 round -5079,8; opt66 2 round. Bỏ 2 round bị cắt ngắn của opt64 (#558: restart 02:13 giữa round ⇒ -5055,9 không có job sâu; #559: round chỉ 4 phút ⇒ -5066,6) thì opt64 8 round trung vị ≈ -5078,4 ≈ opt62 (-5079,6) ⇒ KHÔNG đủ bằng chứng opt64 kém hơn opt62; trung vị 'tệ' gần đây chủ yếu do round bị cắt (restart giữa round, round ngắn 3–5 phút).
- Đính chính ước lượng cũ 'mỗi restart mất ~2 round': trạng thái sàng ĐƯỢC NẠP LẠI sau restart ('screening state restored from /data/calib/screen_state.json'), nên cửa sổ không trống; mất mát chính là round đang chạy (cắt giữa chừng) + thời gian khởi động (~30–60 s) ⇒ ≈1 round/lần restart, ~4–7%… vẫn nên gộp thay đổi.
- Round dài 3–20 phút ⇒ 'nonce tốt nhất mỗi round' phụ thuộc độ dài round; chỉ tiêu công bằng hơn: số nonce ≤ ngưỡng thắng trên mỗi giờ GPU chạy liên tục.

## QUAY VỀ opt62 (02:58 VN 16/11; người dùng: 'opt64 là bạn đổi cách lọc đúng không?')
- opt64 = opt62 + C q2→4, promise -5081→-5079, follow 4→6 (đổi NGƯỠNG lọc, không đổi phương pháp). Số đo khi chạy: opt62 130 phút: 39 job sâu (18,0/giờ), nonce ≤-5083: 6 (2,77/giờ); opt64 77 phút (có ≥3 lần restart): 13 job sâu (10,1/giờ), ≤-5083: 1 (0,78/giờ). Tỉ lệ 3,5×; Poisson P(≤1 | kỳ vọng 3,55) ≈ 0,13 — không chắc chắn thống kê nhưng cùng hướng với quy tắc đã ghi (mọi đổi ngưỡng lọc sau opt51 đều giảm khối thắng) và C q4 làm giảm 44% số job sâu.
- Quyết định: opt62 là bản chính (nhiều dữ liệu, 2 thắng #?; 20:33, 20:59). opt64 được lấy từ mô phỏng (đã nhiều lần lạc quan). Chuỗi thua ≥5 cho phép đổi. KHÔNG đổi ngưỡng lọc nữa nếu không có bằng chứng đo thật.

## MÔ PHỎNG CPU (8 luồng, không GPU; 03:00–03:30 VN 16/11; người dùng yêu cầu 'dùng cpu mô phỏng vài phương án xem có bứt phá không, tối đa 4h, ≤8 luồng')
- Dựng bản CPU C++ của kernel cluster Otsuka (scratch/cpusim/ots.cpp) trên instance thật chain 120 (dump bằng test dump_instances): 4560 nút; 1769 tile K4 gauge (6 trạng thái băng '2 lên 2 xuống'), 1350 cặp ferro, 29241 liên kết yếu |J|=45, h=0. PA như sản xuất (resample 30% mỗi 32 sweep). Hiệu chuẩn: 256x8192 inst0 CPU -5050,7 vs GPU -5043,9 (cùng cỡ). Tốc độ ~21k sweep/s trên 8 luồng.
- THÊM Houdayer ICM (hoán đổi miền liên thông các nút khác nhau giữa 2 bản sao, bỏ miền ≥ n/2; tổng năng lượng bảo toàn) vào quần thể PA. study1 (64 read x 2048 sweep, 4 instance x 2 seed, cùng (instance,seed)): mốc -5013,7; ICM mỗi sweep 1 miền -12,3 (thời gian +34%); mỗi 2 sweep -9,4; mỗi 4 sweep -7,8; mỗi 16 sweep -9,3 (thời gian +0%); mỗi 4 sweep 8 miền -7,5; 32 miền -9,4; mỗi sweep 8 miền -10,0 (+19%). Tham chiếu: GẤP ĐÔI sweep (4096) -15,9 (+105% thời gian); ICM(4,8) ở 4096 sweep -21,0 (thêm -5,1 trên mốc 4096).
- ⇒ ICM gần như MIỄN PHÍ cho ~-8…-12 đơn vị (≈60% hiệu quả của gấp đôi sweep). Ứng viên bứt phá đầu tiên có số đo. Việc kế: kiểm ở quy mô lớn (128x8192, study2) rồi cài kernel GPU (hoán đổi miền giữa các read theo cặp, trong bước PA hoặc kernel riêng).

## opt68 TRIỂN KHAI 03:50 VN 16/11: HOUDAYER ICM TRÊN GPU (kernel quip_cluster_icm; QC_ICM_EVERY mặc định 32) — ỨNG VIÊN BỨT PHÁ ĐẦU TIÊN CÓ SỐ ĐO GPU
- Kernel (cluster.cu): mỗi cặp read (hoán vị ngẫu nhiên mỗi lần) — miền liên thông các nút khác nhau (union-find qua ELL) chọn ngẫu nhiên, bỏ miền ≥ n/2, hoán đổi giữa hai read; tổng năng lượng cặp bảo toàn. device.rs: chạy trước bước xếp hạng PA (mỗi chunk 32 sweep). QC_ICM_EVERY=0 tắt.
- Đo GPU tầng B (256x8192, PA 32/0,3, 100 nonce nhãn, 2 lần mỗi cấu hình, miner dừng 7 phút): không ICM TB -5046,55 / -5047,17 (101 / 100 s); ICM 32: -5050,22 / -5049,30 (103 / 104 s). Cùng nonce: −3,66 (sd 5,05, n=100) ⇒ ~+3,5 đơn vị, chi phí +2–3%; tương quan với deep 0,40/0,48 → 0,48/0,51 (không giảm).
- CPU (ots.cpp, 64x2048): ICM mỗi sweep −12,3; mỗi 16 −9,3 (miễn phí); GPU ở quy mô 8192/pop256 nhỏ hơn (−3,7) — mức lợi giảm khi chạy dài hơn/quần thể lớn hơn; cadence GPU bị chặn ở 32 sweep (chunk).
- opt68 = opt62 (phễu 256:64:32;256:8192:16;512:16384:2;512:32768:4, promise -5081, follow 4, xếp hạng độc lập, state screen_state.json) + ICM. Quay về: opt62.
- Việc kế: đo ICM ở quy mô deep (2048 read 65536 sweep thu nhỏ) và thử ICM mỗi 8/16 sweep (chunk nhỏ hơn); study2 CPU (128x8192) chạy nền.

## KIỂM TRA GHI CHÚ TRƯỚC KHI CHẠY HÀNG ĐỢI CPU QUA ĐÊM (03:55 VN 16/11; người dùng: 'kiểm tra ghi chú trước khi thử, tránh lặp')
- ĐÃ THỬ (bỏ khỏi hàng đợi): quét beta/lịch beta (GPU 10 cấu hình tầng B hôm nay; dòng 104/121/128: beta_max 5–12 phẳng, beta_min>1,5 tệ, hình dạng lịch phẳng); PA mỗi 64 sweep (dòng 424: không nhanh hơn, kém nhẹ) và PA 128; tỉ lệ thay 0,5 (dòng 62/105); đánh đổi read↔sweep thuần (popseed, tradeoff, deeptrade hôm nay); islands, quần thể >2048, PT, reheat, crossover (dòng 125–134).
- CHƯA THỬ (giữ): ICM (chỉ nghi ngờ ở dòng 293, chưa từng đo) và các biến thể: kích thước miền tối đa, nhiều miền/lần, ghép cặp theo năng lượng; PA mỗi 8/16 sweep (dưới ngưỡng đã thử); tỉ lệ thay 0,1/0,7; tương tác ICM×cỡ quần thể (128 read).
- Quy tắc: trước mỗi thử nghiệm mới, grep OPTIMIZATION_NOTES.md các từ khóa của hướng đó; ghi 'đã thử/chưa thử' vào ghi chú trước khi chạy.

## KẾT QUẢ CPU 04:05 VN 16/11 (study2 128x8192; night pha 1 một phần; 4 instance x 2 seed, cùng (instance,seed))
- study2 128 read x 8192 sweep: mốc -5041,5; ICM mỗi 16 sweep −5,5 (thời gian −4%: không tốn); ICM mỗi sweep −5,9 (+7%); ICM mỗi 4 sweep 4 miền −5,9 (−1%). ⇒ ở quy mô GPU lợi ích ~−5,5…−6 và KHÔNG phụ thuộc cadence (mỗi 16 ≈ mỗi sweep) ⇒ cadence 32 trên GPU là đủ (đo GPU −3,7 ở pop 256).
- night pha 1 (64x2048; ICM mỗi sweep = −12,3): cap miền 0,3 −10,1; cap 0,7 −13,3; 4 miền −9,4; ghép cặp theo năng lượng gần nhau −9,9; ghép tốt-nhất-với-tệ-nhất −8,6; **PA mỗi 8 sweep + ICM −16,1** (thêm −3,9 so với ICM+PA32; thời gian +0%…). Chưa biết PA8 một mình (chạy trong night2: base_pa8, base_pa4, icm1_pa4/pa2, cap 0,9/1,0).
- PA mỗi 32 trên GPU hiện đồng bộ host (5–7% chu kỳ): PA mỗi 8 = ×4 chi phí đồng bộ ⇒ cần PA trên GPU (kernel xếp hạng+resample không dtoh) mới đáng; sẽ cân nhắc nếu PA8 cho ≥ −4 ở pop lớn.

## THẮNG #574 (04:16:18 VN 16/11, nonce -5088,635 — SÂU NHẤT từ trước tới nay, opt68=opt62+ICM, 144 s) — khối thắng thứ 3 trong đêm (#552 opt64; #565 opt62; #574 opt68). Mẫu: 1 khối/50 phút opt68, 16 attempt, 7 job sâu, ≤-5083: 1.
## KẾT QUẢ HÀNG ĐỢI CPU (04:35 VN; 8 luồng, không GPU) — chỉ các hướng chưa có trong ghi chú
- 64x2048 (mốc -5013,7; ICM mỗi sweep -12,3): cap miền 0,7/0,9/1,0 → -13,3 (0,9 = 1,0: miền hiếm khi ≥0,7n); cap 0,3 -10,1; 4 miền -9,4; ghép cặp năng lượng gần -9,9; tốt-với-tệ -8,6; PA mỗi 8 một mình -9,8 (PA4 -8,2); ICM+PA16 -13,4; ICM+PA8 -16,1; ICM+PA4 -13,8; ICM+PA2 -12,6; tỉ lệ thay 0,1 -5,8 / 0,7 -8,5 (ICM+PF) kém 0,3 (mặc định) ⇒ PA 32/0,3 gần tối ưu.
- 128x8192 (2 instance x 2 seed; mốc -5042,6): ICM+PA8 -5,3; ICM+PA16 -5,9; ICM(cap0,7) -3,1; ICM+PA8+cap0,9 -6,5 ⇒ lợi thêm của PA dày hơn ICM BIẾN MẤT ở quy mô GPU ⇒ KHÔNG làm PA trên GPU (đóng hướng PA8).
- ICM×quần thể (128x1024): -7,0 (mốc -5001,1 → -5008,1) ⇒ lợi ích ICM giữ khi quần thể lớn.
- Đang chạy night3: ICM ở 512 read x 2048, 256 x 4096, 128 x 32768 (ngoại suy lên deep).

## TRANH CHẤP CPU GIỮA MÔ PHỎNG VÀ MINER (05:05 VN 16/11)
- GPU trung bình 25 mẫu: mô phỏng CPU 8 luồng đang chạy 83% vs tạm dừng 91% (−8 điểm ≈ −9% thông lượng miner, dù đã nice 19; luồng điều khiển GPU nhạy độ trễ). Quy tắc: mô phỏng CPU chạy 4 luồng + nice 19 (người dùng cho phép ≤8). night4.py (SIM_THREADS=4).
- night3 (quần thể × độ dài, 4 instance x 2 seed): 512x2048: mốc -5024,0 → ICM -5034,3 (−10,3); 256x4096: -5034,9 → -5044,6 (−9,7); thời gian CPU +25% / +15%. ⇒ lợi ích ICM KHÔNG giảm theo cỡ quần thể (128x8192 từng −5,9). night4 đo 128x32768 (ngoại suy lên deep).

## ICM GIẢM DẦN THEO ĐỘ DÀI LỊCH Ủ (05:35 VN 16/11; night4 CPU 4 luồng): 128 read x 32768 sweep, 2 instance, 1 seed: mốc -5060,7 → ICM -5061,8 (chỉ −1,1; thời gian +19%); so: 2048 sweep −12; 4096 (256 read) −9,7; 8192 (128 read) −5,9; 32768 −1,1 ⇒ ICM cải thiện tầng NGẮN (A–D), ~0 ở deep (131072 sweep). Khối #574 (-5088,6) KHÔNG do ICM ở deep (có thể do tầng lọc tốt hơn hoặc may).
- Mục tiêu đối thủ (3h, 25 khối): trung vị -5087,18; p25 -5087,97; sâu nhất -5089,30; dốc -0,10/giờ (opponent #581 -5089,1; #583 -5089,3).
- Hướng mới (chưa thử; đã grep ghi chú: single_every/rounds đã thử (dòng 117/118) → không): dùng ICM để RÚT NGẮN sweep ở tầng lọc (B 6144 thay 8192; C 12288 thay 16384) giữ chất lượng, tiết kiệm ~17% GPU ⇒ +20% ứng viên. Đo GPU (icmshort.sh).

## opt69 TRIỂN KHAI 05:47 VN 16/11: ICM RÚT NGẮN SWEEP TẦNG B/C (đo GPU 05:33–05:42; miner dừng 9 phút; chuỗi thua 9 khối ≥5 cho phép)
- Đo (cùng nonce nhãn): B 256x6144 + ICM TB -5046,89 vs B 256x8192 không ICM -5046,70 (−0,19) — thời gian 80 s vs 100 s / 100 nonce (−20%); B 256x6144 không ICM -5042,38 (ICM −4,5). Tương quan với deep: 0,46 / 0,53 / 0,50 (n=100, nhiễu). C 512x12288 + ICM -5056,84 vs C 512x16384 không ICM -5057,09 (+0,24; n=40) — 116 s vs 149 s (−22%); tương quan 0,58 / 0,53.
- opt69 = opt68 (opt62 + ICM mỗi 32 sweep) + QS_STAGES 256:64:32;256:6144:16;512:12288:2;512:32768:4 (D giữ 32768: ICM chỉ −1 ở 32768). Ước tính tiết kiệm GPU: B 20%×42% + C 22%×11% ≈ 11% ⇒ ~+12% ứng viên cùng chất lượng. Quay về: opt68 / opt62.

## SIM CPU: BỎ/ĐỔI TẦNG A TRONG PHỄU opt69 (06:10 VN 16/11; sim_noA.py, 4 luồng nice 19; ρ đo thật, B −20% chi phí, C −22% nhờ ICM) — KẾT QUẢ ÂM, ĐỪNG LẶP LẠI
- T −5086,5: hiện tại (A32 B16 C2 D4) 10,4%/round; bỏ A (qA=1) tốt nhất 10,6%; có A: qA8 15,2%; qA16 14,4%; qA32 15,5%; qA64 16,0%. T −5087,5: hiện tại 7,1%; bỏ A 7,0%; có A tốt nhất 10,7–12,1%. ⇒ GIỮ tầng A (q32). Cùng xu hướng cũ: D q2 và C q4 'tốt' trong mô phỏng nhưng opt64 (C q4) thực tế kém ⇒ không đổi q theo mô phỏng.
- opt69 (05:43→06:02, 19 phút): 5 attempt, 3 job sâu; chưa đủ để đánh giá. Đối thủ: #587 -5089,855 (sâu nhất 3h).

## KIỂM TRA 06:35 VN 16/11: opt69 (0,83 giờ): 8 attempt, 6 sâu (7,2/giờ), ≤-5083: 1 (-5083,7 #589), 0 thắng. opt68 (2,0 giờ): 5,0 sâu/giờ, ≤-5083 1,49/giờ. Chuỗi không thắng 17 khối từ #574. Đối thủ #591 -5091,565 (mới sâu nhất); #587 -5089,9; trung vị 3h ≈ -5087,1…-5087,7 và đang sâu dần (-0,10…-0,2/giờ).

## HƯỚNG MỚI: β_max > 12 VÀ LỊCH HAI ĐOẠN (06:55 VN 16/11; người dùng: 'cần mô phỏng LỚN tìm hướng đột phá; đối thủ đã -5090')
- Lập luận: liên kết yếu |J|=0,045 nhân β=8 = 0,36 ⇒ ở cuối lịch (β=8) lớp yếu vẫn NÓNG (T/J_w ≈ 2,8); lớp mạnh đóng băng từ β~2 nhưng lớp yếu cần β ≳ 20–50. Ghi chú chỉ thử β_max ≤ 12 (dòng 104/128: 5–12 phẳng); β_max > 12 và lịch hai đoạn (kernel có beta_knee/knee_frac) CHƯA thử.
- Thử nhanh 64x2048 inst0 ICM: hi8 -5022,2; hi32 liên tục -5011,0 (kém: ít sweep ở vùng β thấp); knee 8@0,6→64: -5016,9 (kém). Quét có hệ thống ở 128x8192 (night5: hi 8/12/16; knee 8@0,9/0,8 → 16/32; knee 6@0,8 → 24) đang chạy 8 luồng nice 19.

## LỊCH BETA HAI ĐOẠN (07:25 VN 16/11): CPU RẤT TỐT, GPU CHƯA TÁI HIỆN — đang tìm nguyên nhân
- CPU 128x8192 + ICM mỗi sweep, 4 instance x 1 seed (mốc hi8 = -5048,99): β_max 12 liên tục −3,6; hi16 +3,0 (kém); knee 8@0,9→16 −4,65; →32 −5,27; knee 8@0,8→16 −7,35; →32 **−7,83** (3/4 instance −9…−11; 1 instance −1,3). Mức lợi tăng khi dành nhiều sweep hơn cho đoạn β cao.
- GPU (cluster.cu + QC_KNEE_FRAC/QC_KNEE_BETA/QC_BETA_MAX trong sampler.rs ClusterWork::new; B 256x8192 PA 32/0,3 ICM 32; 100 nonce): mốc ICM TB -5049,72; knee0,8→32 -5048,57 (cùng nonce +1,15 = KÉM hơn); knee0,8→16 -5048,38 (+1,34); tương quan với deep 0,38 → 0,52 / 0,57 (không có lợi năng lượng).
- Giả thuyết khác biệt CPU/GPU: (1) ICM mỗi sweep (CPU) vs mỗi 32 sweep (GPU): ICM ở β cao cần dày; (2) β đổi mỗi 32 sweep trên GPU (chunk); (3) số read 256 vs 128. night7 CPU: ICM 32 vs 8 vs 0 với/không knee ở 256x8192.

## LỊCH BETA HAI ĐOẠN: KẾT QUẢ ÂM TRÊN GPU VÀ KHI ICM THƯA (07:40 VN 16/11) — ĐỪNG LẶP LẠI
- GPU (B 256x8192, 100 nonce, cùng nonce so với mốc ICM32 -5049,72): knee 0,8→32 +1,15 (kém); →16 +1,34; ICM mỗi 8 sweep chunk 8: mốc −1,81 (tốt hơn mốc ICM32), knee+ICM8 +0,65 so mốc ICM32; knee+ICM4 −0,06. QC_KNEE_DEBUG xác nhận lịch được áp (hi=200 → −5043,4/−5041,2 vs hi=8 −5041,7/−5046,0, không có lợi).
- CPU 256x8192 ICM mỗi 32 sweep: knee 8@0,8→32 chỉ +1,59 (kém), cặp khớp GPU. Lợi −7,8 trước đó chỉ xuất hiện ở 128 read + ICM MỖI SWEEP; không khả thi trên GPU (chunk 1).
- Phát hiện phụ: ICM mỗi 8 sweep (chunk 8) tốt hơn mỗi 32: −1,8 trên GPU (ICM đã tách khỏi khối PA trong device.rs; QC_ICM_EVERY bội của chunk).

## night8 (07:45 VN 16/11): đánh đổi read↔sweep Ở NGÂN SÁCH SÂU VỚI ICM (đã grep: đánh đổi thuần KHÔNG ICM đã đo trên GPU hôm nay — popseed/tradeoff/deeptrade; tương tác với ICM chưa thử). CPU 8 luồng nice 19: (reads,sweeps) = (128,32768),(256,16384),(512,8192),(1024,4096),(64,65536) cùng tổng 4,2M với ICM mỗi 8 sweep; + mốc không ICM (128,32768),(512,8192). 2 instance x 1 seed. Nếu ICM làm nhiều-read/ít-sweep tốt hơn rõ (≥3) thì thiết kế lại deep (vd 2048 read x 16384 sweep).
- Đã dừng night7 (knee đã đóng).

## GPU DEEP × ICM (08:20 VN 16/11; người dùng: trung vị hiện -5076…-5082 quá thấp; cho phép dừng miner cho mô phỏng >30 phút) — thay night8 CPU bằng GPU (nhanh hơn, đúng thuật toán)
- deepicm.sh: nửa ngân sách deep (shrink 0,35→256, PA 32/0,3, ICM mỗi 8 sweep chunk 8), 18 nonce nhãn; mốc không ICM (deeptrade) 2048x65536 = -5069,78 (26 s/nonce). Cấu hình: 2048x65536; 4096x32768; 8192x16384; 1024x131072. Đánh đổi pop/sweep với ICM chưa thử (đánh đổi thuần không ICM đã âm).

## KẾT QUẢ GPU DEEP × ICM + PT+ICM CPU + CHẨN ĐOÁN CỤM (08:10 VN 16/11) — ÂM, ĐỪNG LẶP LẠI
- GPU (18 nonce nhãn, nửa ngân sách deep, shrink 0,35→256, ICM mỗi 8 sweep chunk 8; mốc 2048x65536 không ICM -5069,78 sd 3,75): 2048x65536 ICM8 −5069,94 (cùng nonce −0,15 ⇒ ICM KHÔNG giúp ở deep); 4096x32768 −5068,74 (+1,04 kém); 8192x16384 −5066,03 (+3,75 kém). ⇒ đánh đổi pop/sweep với ICM không đổi: quần thể lớn + ít sweep tệ hơn.
- PT+ICM CPU (64 replica = 32 nhiệt độ x 2, hoán đổi mỗi 8 sweep, acc 0,42; ICM mỗi sweep; 4096 sweep, inst 0): −4993,7 (β_max 8) / −4986,0 (β_max 16) vs PA+ICM −5035,3 ⇒ PT+ICM kém ~40 đơn vị (khớp kết quả PT cũ âm). ĐÓNG.
- Chẩn đoán bước cụm (CPU, inst 0, 1 read, 4096 sweep): số cụm không tầm thường mỗi lượt 95 → 50; tỉ lệ chấp nhận 44% (β 1,35) → 11% (β 8); kích thước TB 22 → 43; 1 cụm khổng lồ (>10% n) mỗi sweep. ⇒ cụm VẪN hoạt động ở β=8 (không 'đông cứng'); giả thuyết 'chấp nhận cụm sụp ở β cao' bị bác.
- Kết luận chung đến 08:10: các cải tiến ở tầng lọc (ICM +3,5…+5; sweep B/C rút ngắn) đã có; tầng deep bão hòa với mọi trục đã thử (lịch ủ, β_max/knee, read×sweep, ICM, PA, PT).
- GPU deep 1024x131072 ICM8 (18 nonce): -5071,38 (cùng nonce −1,59 vs mốc 2048x65536 không ICM) nhưng 62 s/nonce vs 26 s (×2,4; chunk 8 + ít read kém hiệu quả) ⇒ không đáng. ICM mỗi 8 sweep: B +14% thời gian (120 vs 105 s/100 nonce) cho −1,8 ⇒ ~ngang việc thêm sweep ⇒ KHÔNG triển khai opt70; giữ opt69 (ICM mỗi 32).

## opt70 TRIỂN KHAI 08:35 VN 16/11 (người dùng: 'nâng đuôi lên -5090; gần 2h mới có 1 nonce sâu; đột phá mạnh mẽ') — CHÍNH SÁCH DÒNG DÕI PHỤ (follow-up) NHIỀU HƠN
- Phân tích: sd kết quả deep giữa các entrant (2,9) ≈ nhiễu seed trên MỘT instance (3,1; ghi chú dòng 434) ⇒ entrant gần như cùng μ; kết quả deep ≈ xổ số seed iid. Mỗi lượt follow-up (47 s) là một lần xổ số cùng phân phối với một entrant mới nhưng KHÔNG tốn chi phí tầng lọc (~4–5× deep). Đuôi ∝ số lượt xổ số.
- sim_follow3.py (ρ đo thật, B −20%, C −22% nhờ ICM; ±20% nhiễu): T -5087: hiện tại (promise -5081, follow 4) 10,7%/round; follow 16: 16,5% (×1,6); follow 8: 13,1%; follow 32: 13,9%. T -5090: 2,5% → 4,5% (follow 16, ×1,8). Xu hướng ×1,5–1,9 ở cả hai mức.
- opt70 = opt69 + QS_FOLLOW mặc định 12 (promise -5081 giữ). Chỉ tiêu: số lượt deep+follow mỗi giờ tăng; nonce ≤ -5085/h; đổi nhỏ chỉ một tham số. Quay về: opt69 (ln -sfn /data/quip-cuda-otsuka-opt69).
- Giới hạn trung thực: đuôi ×25 (để -5090 xuất hiện mỗi giờ) cần +4 đơn vị chất lượng; mọi hướng thuật toán đã thử (xem trên) không đạt; follow-up chỉ cho ~×1,5–2.

## NHÁNH TIẾP TỤC TỪ ĐIỂM SHRINK (10:20 VN 16/11; CPU 4 luồng nice 19; ots_br.cpp MODE=branch) — hướng 'xổ số rẻ' (follow-up tiếp tục từ quần thể 35% thay vì chạy lại từ đầu): KẾT QUẢ ÂM
- Cấu hình thu nhỏ: 256 read → 32 @0,35, 4096 sweep, ICM 8, PA 32/0,3. Pha 1 (81% chi phí) chạy một lần; 8 nhánh tiếp tục (19% chi phí mỗi nhánh) với seed khác; so với 6 lần chạy độc lập.
- inst0: nhánh TB -5041,01 sd 1,03 (min -5042,75) | độc lập: -5041,3; -5034,6; -5033,7; -5031,1; -5031,8; -5035,2 (TB -5034,6 sd 3,5). inst1: nhánh TB -5040,12 sd 3,23 (min -5044,07) | độc lập -5036,9; -5040,6; -5039,7; -5036,5; -5036,2; -5035,8 (TB -5037,6 sd 2,0).
- Kết luận: kết quả cuối phụ thuộc chủ yếu vào pha 1 (năng lượng ở 35% tương quan 0,77 với kết quả cuối; nhánh inst0 chỉ sd 1,0): nhánh KHÔNG phải xổ số độc lập, giá trị của nhánh ≈ giá trị pha 1. Follow-up (chạy lại từ đầu) vẫn là cách đúng để lấy lượt xổ số mới; race-abort sớm (đã có QC_RACE ở 25%) mới là đòn bẩy rẻ để bỏ lượt xấu. Hướng 'warm start/nhánh' ĐÓNG.

## PHÁT HIỆN: TẦNG A CHỈ CÓ 2 BẬC BETA (10:55 VN 16/11) — đang đo GPU (aspb.sh)
- Mã (sampler.rs ClusterWork::new): sweeps_per_beta mặc định = QC_SPB=32 ⇒ tầng A (64 sweep) chỉ có 64/32 = 2 bậc beta: 32 sweep ở β=1,2 rồi 32 sweep ở β=8 (không ủ mượt). Các tầng B/C/D/deep có hàng trăm bậc nên không bị.
- CPU (judgeA.txt; 100 instance thật, 256x64 ủ MƯỢT 64 bậc, không ICM): TB best -4876,5; tương quan với deep1 0,41 (best) / 0,42 (mean-read), với B 0,45. Trên GPU A (2 bậc) TB -4834…-4844 và tương quan với deep chỉ 0,10 (ma trận tương quan 437/100 nonce). ICM không cải thiện A (0,28…0,42, nhiễu). Cấu hình 128x128: 0,28/0,50; 64x256: 0,28/0,33; 32x512: 0,28/0,29 (best/mean-read).
- Giả thuyết: ủ mượt A (spb nhỏ) nâng ρ(A,deep) 0,10 → ~0,4 với chi phí thêm (nhiều launch) ⇒ bộ lọc đầu chọn instance tốt hơn rõ. Đo GPU: A 256x64 với SPB 32/8/4/2/1 trên 100 nonce nhãn, tương quan với deep1.

## opt71 TRIỂN KHAI 11:00 VN 16/11: TẦNG A ỦM MƯỢT (sweeps_per_beta 2 thay vì 32) — ĐO GPU 10:54
- GPU, A 256x64, 100 nonce nhãn, không ICM: SPB 32 (2 bậc beta) TB -4837,7, tương quan với deep1 0,29, với B 0,16; SPB 8: -4880,0 / 0,25 / 0,27; SPB 4: -4879,3 / 0,33 / 0,28; SPB 2: -4877,8 / 0,35 / 0,26; thời gian cả 100 nonce: 7 / 6 / 6 / 6 s (không tốn thêm). (SPB=1 bị coi là 'không đặt' ⇒ dùng 32.) Năng lượng A tốt hơn 42 đơn vị; tương quan tăng +0,05…+0,06 (deep, nhiễu ±0,09) và +0,10 với B.
- opt71 = opt70 + tầng A (sweeps_k ≤ 128) dùng QC_SPB_A mặc định 2. Cửa sổ A lưu trong screen_state.json chứa năng lượng cũ (~-4840): năng lượng mới (~-4880) đều 'tốt hơn' ⇒ A cho qua hết cho tới khi cửa sổ 256 phần tử thay xong (~vài chục giây). Quay về: opt70.
- opt71 xác nhận trên miner thật (11:04): năng lượng A trong calib -4884,8 (trước -4840: mượt tốt hơn ~42 đơn vị). BÀI HỌC: đổi PHÂN PHỐI năng lượng của một tầng làm cửa sổ xếp hạng lưu (screen_state.json) sai ⇒ A cho qua 100% (mọi năng lượng mới 'tốt hơn' cửa sổ cũ) ⇒ mọi salt xuống B (0,9 s) ⇒ thông lượng salt giảm 2100 → 9/s trong ~7 phút (mất ~2 round) cho tới khi 256 phần tử cửa sổ A thay xong. Lần sau: xóa phần tử cửa sổ tầng bị đổi trong screen_state.json (hoặc dùng tên tệp trạng thái mới) TRƯỚC khi restart.

## KIỂM TRA GPU RẢNH (11:35 VN 16/11; 1200 mẫu 2 Hz, opt71 ổn định) — ÂM: không có khoảng rảnh để lấy lại
- mức dùng GPU TB 86%; <50%: 1,4% số mẫu; <20%: 0,7%; khoảng rảnh <30%: 2 lần, dài nhất 4,0 s (tổng 4 s/600 s); công suất TB 385 W. Thời gian host mỗi phút (init ~3,5 s; launch_sync ~52 s; pa_host ~3 s) tương đương opt69/70 ⇒ opt71 không làm chậm. ⇒ GPU đã bận; thông lượng chỉ tăng được bằng thuật toán/kernel, không bằng lấp khoảng rảnh.
- opt71 (0,55 giờ): 10 attempt, 6 sâu (11,0/giờ), tốt nhất -5080,6; thông lượng salt phục hồi 1004 → 1847/s.

## PHÂN RÃ NĂNG LƯỢNG NGHIỆM TỐT NHẤT (12:00 VN 16/11; CPU ots_parts.cpp, 128x8192 ICM8, inst 0/1)
- inst0 -5052,9 = lớp mạnh -4868,0 + lớp yếu -184,9; inst1 -5047,8 = -4868,0 + -179,8. Lý tưởng lớp mạnh (mọi tile 2-lên-2-xuống, mọi cặp thỏa) = -4888,0 ⇒ nghiệm tốt nhất VI PHẠM 6 tile + 4 cặp (inst0) / 4 tile + 6 cặp (inst1): trả ~20 đơn vị lớp mạnh để lấy lớp yếu. Tối ưu thật là đánh đổi có khuyết tật (defect), không nằm trọn trên đa tạp băng. Chưa khai thác được (gợi ý: nước đi sinh/hủy cặp khuyết tật theo trường yếu).
## QUY TRÌNH MỚI (người dùng nhắc: không dừng sớm bản xấu; quên chạy mô phỏng): canary.sh sau mỗi deploy (hủy sau ≤15 phút nếu thông lượng/lỗi/GPU xấu) + kiểm chất lượng ngoại tuyến trước deploy.

## QUAY VỀ opt68 (11:55 VN 16/11; người dùng: 'chạy bản nào để khôi phục tỉ lệ thắng và nâng trung vị?')
- Bảng hiệu quả theo bản (giờ chạy thật; từ attempts + API khối): opt44–51 8,9 h: 1,01 thắng/h, ≤-5083 1,34/h, trung vị round -5080,8 | opt52–59 (msa/S0 cao) 5,5 h: 0 thắng, 0,36/h, -5080,0 | opt61/62 2,3 h: 1,32 thắng/h, 2,65/h, -5080,6 | opt62 (sau) 1,6 h: 0,65/h, 1,94/h, -5079,2 | opt64 1,3 h: 0,77/h, 0,77/h, -5071,7 | opt65 1,0 h: 0, 1,03/h, -5079,8 | opt68 1,9 h: 0,54/h, 1,62/h, -5079,7 | opt69 1,9 h: 0, 1,03/h, -5076,4 | opt70 2,5 h: 0,40/h, 0,40/h, -5075,7 | opt71 1,2 h: 0, 0/h, -5074,3.
- XU HƯỚNG ĐƠN ĐIỆU XẤU từ opt68 → opt71 (trung vị round -5079,7 → -5076,4 → -5075,7 → -5074,3; ≤-5083/h 1,62 → 1,03 → 0,40 → 0): các thay đổi opt69 (B 6144/C 12288), opt70 (follow 12), opt71 (A mượt) KHÔNG có lợi trên miner thật dù có số đo ngoại tuyến. Mẫu nhỏ/nhiễu nhưng cùng chiều ⇒ quay về bản cuối chưa có xu hướng xấu: opt68 (= opt62 + ICM32, đã có khối #574). Fallback canary: opt62.
- Triển khai: dừng container, dời screen_state.json → screen_state.opt71.json.bak (tránh cửa sổ A ủ mượt -4880 làm A chặn hết khi quay về A 2 bậc -4840), restart opt68, canary.sh opt68 opt62 15.

## MÔ PHỎNG CPU: TÌM KIẾM CỤM ĐA-SPIN CHÍNH XÁC (có thể sinh khuyết tật) TRÊN NGHIỆM CUỐI (12:10 VN 16/11; ots_pol.cpp) — ÂM
- 8 read tốt nhất của chạy 128x4096 (ICM8), mỗi read 400000 lần thử cụm liên thông ngẫu nhiên (kích thước 2–12), đảo nếu ΔE tổng (kể cả lớp mạnh) <0: inst0 TB −0,44 đơn vị (22 cụm được chấp nhận tổng cộng), inst1 −0,39. Nghiệm cuối có 4–8 tile + 4–6 cặp vi phạm (trả ~12–28 ở lớp mạnh) nhưng nước đi cục bộ không cải thiện thêm đáng kể ⇒ khuyết tật đã tối ưu cục bộ (khớp kết quả DP chính xác cửa sổ ≤400 nút = 0 cải thiện). Hướng 'nước đi khuyết tật' ĐÓNG; khoảng cách còn lại là chọn BỒN toàn cục, không phải tối ưu cục bộ.

## SIMULATED BIFURCATION (SB) TRÊN CPU (12:25 VN 16/11; sb.cpp) — ÂM: kém ~70 đơn vị; KHÔNG viết kernel SB
- inst0, 32 replica, 2048 bước (cùng chi phí ~ PA+ICM 64x2048 = -5026): dSB c0f 0,5/1/2/4: best -4858/-4894/-4931/-4724; bSB c0f 1/2: -4953/-4919. Tốt nhất -4953 vs PA+ICM -5026…-5035 (kém ~70–80). Chạy 8× nhiều bước hơn (16384): xem kết quả bên dưới. Lý do hợp lý: ràng buộc băng 2-lên-2-xuống (tile K4) làm cảnh quan không trơn; SB liên tục không giữ đa tạp băng.
- Ứng viên 'kernel khác' còn lại (chưa mô phỏng): (1) LNS với bộ giải chính xác (SAT/ILP) trên lân cận KHÔNG gọn (hợp các vòng dài) — DP cửa sổ gọn ≤400 nút và pairworm upstream đều 0 cải thiện nên rủi ro cao; (2) giải mã lan truyền niềm tin (BP) dẫn hướng + PA — chưa có bằng chứng.

## SQA (ỦA LƯỢNG TỬ MÔ PHỎNG / PIMC) TRÊN CPU (12:50 VN 16/11; sqa.cpp; người dùng: 'có thử nghiệm mới mới có tiến bộ') — ÂM
- Inst0, M=8 chuỗi, P lát Trotter, J_perp theo Γ giảm 2→0,005, nước đi Metropolis một spin + đảo cả đường thế giới của spin. Cùng chi phí ~131k sweep-tương-đương: P8 S2048 β8 G0 2: best -4990,5 (mean -4980,8); β6 G0 3: -4988,4; β12: -4979,5; P4 S4096: -4944,7. Chi phí 32k: -4941…-4921. Tham chiếu PA+ICM 64x2048 (131k): -5026 ⇒ SQA kém ~35–45 đơn vị; đúng xu hướng 'bước một spin không đi qua ràng buộc băng'. ĐÓNG hướng SQA.
- Tổng kết họ thuật toán đã đo trên instance thật (CPU/GPU): PA+cụm FK+ICM (tốt nhất), PT -40, PT+ICM -40, SB -50…-80, SQA -35…-45, worm/DP/cụm đa-spin ~0.

## CỤM FK VỚI CHỌN LIÊN KẾT TILE CÓ ĐỊNH HƯỚNG THEO TRƯỜNG YẾU (13:05 VN 16/11; ots_guide.cpp GUIDE=κ) — ÂM
- Thay lựa chọn ngẫu nhiên đều của liên kết (+/−) trong tile 2-lên-2-xuống bằng xác suất ∝ exp(−κβ·ΔE_yếu) (ΔE_yếu = thay đổi năng lượng lớp yếu nếu đảo cặp nút đó); không sửa cân bằng chi tiết (chỉ tối ưu). 64x2048, ICM8, 4 instance, seed 1, TB năng lượng: κ=0 −5023,2; κ=0,25 −5020,3; κ=0,5 −5021,7; κ=1,0 −5018,8 ⇒ KHÔNG cải thiện (kém 1,5–4,4). ĐÓNG. (Chọn liên kết ngẫu nhiên đều đã đủ; thông tin trường yếu không giúp ở bước dựng cụm vì quyết định chấp nhận cụm đã dùng trường yếu.)

## CẤU TRÚC CUỘC ĐUA THEO THỜI GIAN (13:20 VN 16/11; từ /api/telemetry recentDifficulty + log coordinator)
- Ngưỡng (trần năng lượng, loại trừ) NỚI 0,125 đơn vị mỗi block (6 s) = ~1,25 đơn vị/phút trong một round (vd 39648 -5087,21 → 39653 -5086,585). Người thắng = người đầu tiên TÌM được nonce thấp hơn trần hiện tại (năng lượng thắng thường sát trần, thỉnh thoảng sâu hơn). Tỉ lệ nonce ≤ x tăng ×2,2 mỗi +1 đơn vị ⇒ mỗi phút trần nới ×~3 tốc độ xuất hiện nonce hợp lệ ⇒ round kết thúc khi tích phân tốc độ ≈ 1. P(thắng) ∝ tốc độ tìm nonce ≤ trần của ta so với đối thủ: thiếu 1 đơn vị chất lượng = chia 2,2 cơ hội.
- Thua sát: #640 (05:32): ta -5086,385 vs trần lúc đối thủ nộp (05:32:17) -5086,46 (thiếu 0,075; block kế -5086,335 sẽ nhận) — đối thủ -5089,0 vào trước ~0,5 s. #635: -5086,0 vs -5086,5. Coordinator đã có stash + mô hình decay (stash.rs/decay.rs) chiếu block khả thi; thắng thua sát là đua độ trễ nộp, không còn đòn bẩy rõ.

## ĐÍNH CHÍNH 13:35 VN 16/11 (người dùng: 'để thắng cần nonce -5090…-5092?'): độ sâu cần để thắng
- Năng lượng thắng của 120 khối gần nhất (API /api/blocks): trung vị −5086,9; p25 −5087,8; p75 −5086,0; sâu nhất −5091,6; chỉ 4/120 ≤ −5089 (3%). Theo thời gian round: 0–120 s −5086,5; 120–240 −5087,1; 240–360 −5086,6; 360–600 −5087,1; 600–900 −5086,3 (round trung vị 315 s) — KHÔNG phụ thuộc rõ vào độ dài round.
- Ghi chú 13:20 nói trần nới 1,25 đơn vị/phút trong round chỉ quan sát được trong cửa sổ 50 block của /api/telemetry; chuỗi /api/difficulty-history thưa (−5086,0…−5087,7 qua nhiều giờ) ⇒ KHÔNG xác nhận nới liên tục; năng lượng thắng không tăng theo thời gian round. Đừng dùng 'nới 1,25/phút' để lập luận chiến lược.
- Hệ quả: độ sâu cần để thắng ~−5086,5…−5088 (giữa phân phối), −5090 chỉ ~3%. opt68: nonce tốt nhất mỗi round −5086,4 (tối đa) trong 8 round; chưa đạt trung vị thắng (−5086,9) ở round nào ⇒ cần dịch phân phối nonce sâu thêm ~+1…+1,5 đơn vị (khớp ước lượng trước).

## KIỂM TOÁN DEEP (16/11 13:57–14:12 VN, miner dừng 15 phút, GPU, harness_knee3, deep production 2048×131072, PA 32/0,3, ICM32, shrink 0,35→256)
Chạy lại 16 nonce live đã có kết quả deep thật. Deep thật tb −5080,60 (tốt nhất −5084,5); tái chạy tb −5078,81 (tốt nhất −5083,1); chênh +1,78 (nông hơn) sd 2,82 ⇒ ~2,5σ, tương quan thật–tái chạy 0,39. Kết luận: harness tái hiện deep production trong ~1,8 (phần lớn do hồi quy về trung bình/seed), deep KHÔNG hỏng rõ ràng; nhiễu seed sd ~2,8 ≈ độ phân tán giữa instance ⇒ lọc chọn instance chỉ giải thích ~40% phương sai deep (khớp ghi chú trước). Chưa kiểm: có tầng lọc nào loại nonce sâu (cần mẫu 'deep không qua lọc', tốn GPU).

## ABLATION TỪNG TẦNG (16/11 14:32–15:07 VN, GPU, miner dừng 34 phút; harness_knee3, 32 nonce NGẪU NHIÊN không qua lọc, 31 có đủ dữ liệu)
Tầng: S1 256×64, S2 256×8192, S3 512×16384, S4 512×32768, deep 2048×131072 (PA 32/0,3, ICM32, shrink 0,35→256). Năng lượng trung bình (sd): S1 −4837,8 (7,1); S2 −5049,2 (3,2); S3 −5060,0 (3,8); S4 −5063,7 (4,1); deep −5071,8 (3,3).
Tương quan với deep: S1 −0,11 (vô dụng ở 64 sweeps), S2 0,39, S3 0,47, S4 0,44; S2–S3 0,41, S3–S4 0,30.
Top-8/31 theo tầng → deep trung bình (tất cả −5071,8): S1 −5070,9; S2 −5073,7; S3 −5074,0; S4 −5073,4 (tốt hơn ~2 đơn vị). Nonce deep sâu nhất xếp hạng #19 theo S1, #7 theo S2, #2 theo S3, #3 theo S4.
Kết luận: (1) deep nhất quán, không lỗi; (2) tầng S1 ngắn 64 sweeps không dự báo gì trên nonce ngẫu nhiên (nhưng ở production S0 + A còn tác dụng ở mức khác, chưa đo riêng); (3) S2–S4 dự báo vừa phải (r≈0,4–0,5), không tầng nào loại hệ thống nonce sâu (nonce sâu nhất vẫn ở top-3 theo S3/S4); trần của phễu là nhiễu seed deep (sd 3,3 ≈ sd giữa instance). Mẫu nhỏ (n=31), sai số r ~±0,17.

## DEEP NHIỀU SEED TRÊN CÙNG INSTANCE (16/11 15:59–16:42 VN, GPU, miner dừng 43 phút): 16 nonce production × 4 seed (5,11,12,13), deep 2048×131072
- Nhiễu seed NỘI BỘ (cùng instance) sd 1,62 (không phải 2,8–3,3 như ước lượng trước từ chênh deep thật–tái chạy); sd giữa trung bình-instance 1,96 (mẫu đã qua lọc nên hẹp). Trung bình mỗi seed −5078,8…−5079,9 (ổn định). Tương quan deep production vs trung bình 4 seed: 0,43.
- Best-of-4 seed trên cùng nonce: tb −5080,9 so với 1 lần −5079,3 (sâu hơn 1,57 với 4× chi phí). 64 lần chạy deep: tốt nhất −5083,6, KHÔNG lần nào ≤ −5085; lần đơn ≤ −5083: 3,1%, ≤ −5085: 0%.
- 4 nonce tốt nhất theo trung bình 4 seed có deep tb −5081,8 (vs −5079,3 chung): chọn đúng instance +2,5; chạy 4 seed trên 4 nonce tốt nhất cho tốt nhất −5083,1 = bằng 16 nonce × 1 seed.
- Kết luận: với chất lượng instance đã qua lọc, phân phối deep của thuật toán gần như KHÔNG chạm −5085 (0/64); mốc thắng −5087 nằm ngoài đuôi ~3,5 đơn vị. Thêm seed không thắng được lọc; khoảng cách là chất lượng thuật toán giải (đuôi phân phối), không phải chiến lược chọn/lặp. Hướng 'nhiều lineage trên instance tốt' = KHÔNG hiệu quả hơn hiện tại.

## night9 CPU (16/11 17:10–17:32 VN): LỊCH HAI ĐOẠN × ICM MỖI SWEEP, 256x8192 — ÂM, ĐÃ HỦY SỚM
- ICM mỗi sweep, hi8 (β_max 8): inst0 −5057,9; inst1 −5058,6; inst2 −5051,6; inst3 −5050,3 (best). Cùng ICM mỗi sweep + knee 8@0,8→32: inst0 −5049,9 (KÉM 8,0); inst1 −5054,2 (KÉM 4,5). Tiêu chí hủy (cần sâu hơn ≥3) không đạt → dừng, bỏ e4.
- ⇒ lợi −7,8 của night5 (128 read, ICM mỗi sweep) KHÔNG tái hiện ở 256 read; lịch hai đoạn kém cả khi ICM mỗi sweep. Giả thuyết 'kernel ICM hợp nhất mỗi sweep' bị bác. ĐỪNG LẶP LẠI. (Phụ: ICM mỗi sweep với hi8 cho −5057,9 vs ICM32 ~−5050 ở inst0 nhưng tốn ×~2 thời gian CPU.)

## ĐÍNH CHÍNH night9 (18:02 VN): 'ICM mỗi sweep sâu hơn ICM32 ~8' là SAI (chọn instance 0). Trung bình 4 instance (best, 256x8192 hi8): ICM32 (night7) −5055,0 [−5052,2; −5054,8; −5059,8; −5053,3] vs ICM mỗi sweep (night9) −5054,6 [−5057,9; −5058,6; −5051,6; −5050,3] ⇒ NGANG NHAU (chênh 0,4, trong nhiễu), ICM mỗi sweep tốn ×1,8 CPU. Kernel ICM hợp nhất KHÔNG có cơ sở. night10 (so cùng chi phí) hủy vì tiền đề sai.
