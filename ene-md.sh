#!/bin/bash -ex

for l in 1e-1 1e-2 1e-3
do 
  for b in 0.1 1 10 50 100 500 1000
  do
    for c in 0.001 0.01 0.1 1 10
    do
      for r in 0.2 0.4 0.6 0.8 1 2 4
      do
        while true
        do
          count=$(ps -ef | grep -c kk_test)
          if [ $count -lt 2 ]
            then
            # 改动项 查询第1块gpu的容量--2p 第2块3p--2  第三块--4p  第四块--5p +2
            stat2=$(gpustat | awk '{print $11}' | sed -n '2p')
            if [ "$stat2" -lt 6000 ]
              then
                echo 'run'
                #改动项 前面输入占用的gpu id 后面是运行代码
                python clmd.py --DS ENZYMES --lr 0.01 --local --num-gc-layers 3 --aug random2 --seed 0 -DS_pair ENZYMES+PROTEINS -batch_size_test 128 --device 0 --amplr $l --bal1 $b --bal2 $c --ratio $r 
                sleep 5
                break
            fi
          fi
        done
      done
    done
  done
done
