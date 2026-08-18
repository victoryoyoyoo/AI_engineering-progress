# Review Queue

Things Claude simplified/skipped during teaching, logged so I can come back and fill them in later. Cross-check against the "Review queue" section in `LEARNING.md` too — that one tracks quiz/objective gaps, this one tracks specific deferred items with the reason each was deferred.

## Lesson 1 — Linear Algebra Intuition (2026-08-18)

標準：延後的理由是「跟目標（打造AI產品）相不相關」，不是「難不難」。以下逐項寫清楚。

- **normalize**（向量正規化，縮放成長度1）— 講過原理沒手打進 practice.py。關聯度中低：主要用在 embedding 正規化，但不是每個應用都需要，之後真的做 RAG/embedding 相關功能時可能會用到。
- **angle_between**（算兩向量夾角度數）— 講過原理沒手打。關聯度低：實務上比相似度都是直接比較 cosine_similarity 的數值，很少真的需要轉換成角度。
- **project_onto**（投影）— 講過原理沒手打。關聯度中低：主要是 Gram-Schmidt 的內部工具，單獨用在 AI 產品開發的機會不多，但線性迴歸/PCA 背後有用到這個概念。
- **is_independent**（線性獨立判斷）— 理解邏輯沒手打。關聯度低：偏數學理論驗證，日常 AI 產品開發很少直接呼叫這個。
- **gram_schmidt**（正交化）— 理解邏輯沒手打。關聯度低：用在數值方法/QR分解這類進階數值計算，不是典型 AI 應用開發會直接手刻的東西。
- **rank**（矩陣的秩）— 只提過名詞，測驗答錯的就是這題。**老實說這個關聯度其實不低**（LoRA 用到 rank 的概念），今天沒教好主要是時間排序問題，不是真的不相關，之後應該優先補。
- **transpose**（矩陣轉置）— Matrix class 簡化時整個拿掉了，完全沒教。關聯度中等：矩陣基本操作之一，之後處理資料形狀轉換會用到，今天純粹是為了縮減 Matrix 篇幅而砍掉，跟目標相關性判斷關係不大。
- **Matrix 乘矩陣**（matrix @ matrix，3層迴圈版本）— 完整版本在 `01-math-foundations/01-linear-algebra-intuition/reference.py`，只追蹤過邏輯沒手打進 practice.py。關聯度中高：多層神經網路運算會用到，比矩陣乘向量進階一階，之後有餘力應該補手打。
- **Julia 版本** — 完全沒做，刻意跳過。關聯度低：不同程式語言，跟主力學 Python 打造 AI 產品的路線不直接相關。
- **PyTorch 版本、QR 分解、Ship It、官方 Exercises** — 完全沒做。PyTorch 關聯度高（之後深度學習會大量用到，只是這一課只是入門展示，之後課程會有專門的 PyTorch 內容）；QR 分解關聯度中低（進階數值方法）；Ship It 是寫文字說明文件不是寫程式碼，關聯度低；官方 Exercises 內容跟已經練過的方法重疊度高，關聯度低。

## 已放棄

Phase 回頭日那天做不完、正式決定不補的項目會移到這裡，目前還沒有（Phase 1 還沒做完，還沒到回頭日）。
