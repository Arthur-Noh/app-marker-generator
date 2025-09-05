# SK일렉링크 마커 생산 프로그램

### 원하는 운영사만 생성하기
```angular2html
> OP=st

# 전수 생성: selected/unselected × lp/no-lp × fast 00..99 × slow 00..99
for sel in selected unselected; do
  for lp in lp no-lp; do
    for f in $(seq 0 99); do
      for s in $(seq 0 99); do
        python src/generator.py --operator "$OP" --type detail_long \
          --selected "$sel" --lp "$lp" --fast "$f" --slow "$s"
      done
    done
  done
done
```