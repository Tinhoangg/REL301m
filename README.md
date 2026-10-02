# Reinforcement Learning for Go 9x9

Project này cài đặt một agent chơi cờ vây 9x9 bằng **Self-Play**, **Monte Carlo Tree Search** và mạng **Policy-Value** nhỏ bằng PyTorch.

## 1. Environment

Environment nằm trong file `go_env.py`, class chính là `GoEnv`.

Environment chịu trách nhiệm quản lý:

- bàn cờ Go 9x9;
- lượt đi của đen/trắng;
- nước đi hợp lệ và không hợp lệ;
- bắt quân;
- suicide move;
- pass;
- kết thúc ván khi hai bên pass liên tiếp;
- tính điểm theo Tromp-Taylor area scoring;
- reward cuối ván.

Khởi tạo:

```python
from Go_rel.go_env import GoEnv

env = GoEnv(size=9, komi=7.5)
state = env.reset()
```

Mỗi bước chơi:

```python
next_state, reward, done, info = env.step(action)
```

Trong đó:

- `next_state`: state mới sau khi đi;
- `reward`: điểm thưởng;
- `done`: ván cờ đã kết thúc chưa;
- `info`: thông tin phụ như player, captured, score, winner.

## 2. State

State là trạng thái bàn cờ hiện tại mà agent nhìn thấy.

Trong code có 2 dạng state:

### Board gốc

`env.board` là ma trận kích thước `9 x 9`.

Giá trị trên board:

```text
 0  = ô trống
 1  = quân đen
-1  = quân trắng
```

Ví dụ:

```python
print(env.board)
```

### Encoded state cho neural network

`env.encode()` trả về tensor NumPy có shape:

```text
(3, 9, 9)
```

Ba plane gồm:

```text
plane 0: quân của người đang tới lượt
plane 1: quân của đối thủ
plane 2: màu lượt hiện tại
```

Với plane 2:

```text
1.0 = đen đang tới lượt
0.0 = trắng đang tới lượt
```

Lý do encode theo góc nhìn người đang đi: model có thể học chung một chiến lược cho cả đen và trắng.

## 3. Action

Action là một số nguyên.

Với bàn cờ 9x9:

```text
0 đến 80 = đặt quân lên một giao điểm
81       = pass
```

Mapping action sang tọa độ:

```python
row = action // 9
col = action % 9
```

Ví dụ:

```text
action 0  -> row 0, col 0
action 10 -> row 1, col 1
action 40 -> row 4, col 4
action 81 -> pass
```

Trong code:

```python
action = env.coord_to_action(row, col)
row, col = env.action_to_coord(action)
```

Danh sách nước đi hợp lệ:

```python
legal_actions = env.legal_actions()
```

Kiểm tra một action có hợp lệ không:

```python
env.is_legal(action)
```

Nước đi không hợp lệ gồm:

- action ngoài khoảng;
- đặt quân vào ô đã có quân;
- suicide move không bắt được quân đối thủ;
- vi phạm positional superko.

## 4. Reward

Reward hiện tại được thiết kế đơn giản:

```text
trong ván: 0.0
cuối ván nếu thắng: +1.0
cuối ván nếu thua: -1.0
hòa: 0.0
```

Reward chỉ xuất hiện rõ ràng khi ván kết thúc. Trong các nước đi giữa ván, reward là `0.0`.

Trong `env.step(action)`, reward được tính theo góc nhìn của người vừa đi:

```python
if winner == player_vua_di:
    reward = +1.0
else:
    reward = -1.0
```

Trong self-play, sau khi biết winner, code gán value target cho mỗi state:

```text
+1 nếu player tại state đó là người thắng
-1 nếu player tại state đó là người thua
 0 nếu hòa
```

## 5. Scoring

Project dùng **Tromp-Taylor area scoring**.

Điểm đen:

```text
số quân đen trên bàn + vùng trống chỉ bị đen bao quanh
```

Điểm trắng:

```text
số quân trắng trên bàn + vùng trống chỉ bị trắng bao quanh + komi
```

Mặc định:

```text
komi = 7.5
```

Lấy điểm:

```python
score = env.score()
winner = env.winner()
```

`winner` trả về:

```text
 1  = đen thắng
-1  = trắng thắng
 0  = hòa
```

## 6. Agent



Environment chỉ quản lý luật chơi:

```text
state + action -> next_state + reward + done
```

Agent là phần chọn action dựa trên state.

Trong project này agent gồm:

- Policy-Value Network trong `model.py`;
- MCTS trong `mcts.py`;
- Self-play loop trong `self_play.py`.

## 7. Policy

Policy là chiến lược chọn nước đi.

Với Go 9x9, policy là vector dài `82`:

```text
policy[0]  = xác suất chọn action 0
policy[1]  = xác suất chọn action 1
...
policy[80] = xác suất đặt vào ô cuối cùng
policy[81] = xác suất pass
```

Tổng các xác suất bằng `1.0`.

Trong self-play, policy target không phải policy raw của network, mà là policy cải thiện từ MCTS:

```text
policy target = visit counts của MCTS sau khi search
```

## 8. Value

Value là dự đoán trạng thái hiện tại tốt hay xấu.

Mạng neural network trả về:

```text
value trong khoảng [-1, 1]
```

Ý nghĩa:

```text
value gần +1 = người đang tới lượt có khả năng thắng cao
value gần -1 = người đang tới lượt có khả năng thua cao
value gần  0 = chưa rõ / cân bằng
```

## 9. Model

Model nằm trong `model.py`, class `PolicyValueNet`.

Input:

```text
(batch, 3, 9, 9)
```

Output:

```text
policy_logits: (batch, 82)
value:         (batch,)
```

Model có hai head:

- policy head: dự đoán nước đi nên chọn;
- value head: dự đoán khả năng thắng/thua.

Loss khi train:

```text
loss = policy_loss + value_loss
```

Trong log:

```text
epoch 1/10 | loss 4.4840 | policy 4.1723 | value 0.3118
```

Ý nghĩa:

- `policy`: lỗi khi model bắt chước policy từ MCTS;
- `value`: lỗi khi model dự đoán kết quả thắng/thua;
- `loss`: tổng của hai lỗi trên.

## 10. Monte Carlo Tree Search

MCTS nằm trong `mcts.py`.

MCTS dùng model để:

- lấy prior policy cho các action;
- lấy value estimate cho state;
- mở rộng cây tìm kiếm;
- chọn action dựa trên visit counts.

Số simulations càng lớn thì agent suy nghĩ lâu hơn nhưng thường tốt hơn.

Ví dụ:

```powershell
python -m evaluate --games 10 --simulations 50
```

`--simulations 50` nghĩa là mỗi nước đi MCTS chạy 50 lần mô phỏng.

## 11. Self-Play

Self-play là quá trình agent tự chơi với chính nó để sinh dữ liệu train.

Mỗi state trong ván sẽ lưu:

```text
state
policy từ MCTS
value target từ kết quả cuối ván
```

Chạy self-play:

```powershell
python -m self_play --games 50 --simulations 50
```

File dữ liệu mặc định:

```text
data/self_play_latest.npz
```

Trong file này có:

```text
states
policies
values
winners
game_lengths
```

## 12. Training

Train model từ dữ liệu self-play:

```powershell
python -m train --epochs 10
```

Mặc định đọc:

```text
data/self_play_latest.npz
```

Mặc định lưu checkpoint:

```text
checkpoints/latest.pt
```

Xem nhanh các game trước khi train:

```powershell
python -m train --epochs 10 --replay-games --replay-delay 0.05
```

## 13. Evaluation

Đánh giá agent với random baseline:

```powershell
python -m evaluate --games 10 --simulations 50
```

Kết quả in ra:

```text
wins
losses
draws
win_rate
```

## 14. GUI

Mở giao diện chơi với agent:

```powershell
python -m play_gui
```


Cho hai agent tự chơi với nhau trên GUI:

```powershell
python -m play_gui --auto-play --simulations 10 --auto-delay 300
```

Trong GUI:

- `Human`: người chơi với agent;
- `Auto`: hai agent tự chơi;
- `Simulations`: số lần MCTS search mỗi nước;
- `Auto delay ms`: tốc độ xem hai agent tự đánh.

## 15. Các lệnh thường dùng

Chạy test:

```powershell
python -m unittest discover Go_rel/tests
```

Sinh dữ liệu:

```powershell
python -m self_play --games 50 --simulations 50
```

Train:

```powershell
python -m train --epochs 10
```

Evaluate:

```powershell
python -m evaluate --games 10 --simulations 50
```

Chơi terminal:

```powershell
python -m play_cli --simulations 50
```

Chơi GUI:

```powershell
python -m play_gui --simulations 25
```


Pipeline hiện tại:

```text
environment -> MCTS -> self-play -> training -> evaluation -> GUI
```

```text
self-play -> train -> self-play bằng model mới -> train tiếp -> evaluate
```

Ví dụ:

```powershell
python -m self_play --games 100 --simulations 100
python -m train --epochs 10
python -m evaluate --games 20 --simulations 100
```

