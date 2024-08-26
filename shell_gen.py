import sys
import os

def main():
    if len(sys.argv) != 4:
        print("Usage: python script.py <a> <b> <c>")
        sys.exit(1)
    
    a = int(sys.argv[1])  # 每个section有几行
    b = int(sys.argv[2])  # 每个CSV出现的次数
    c = int(sys.argv[3])  # CSV序列结束的数字

    output_dir = 'markets'
    script_filename = 'BSE.py'
    sh_filename = 'simulator/sub.sh'


    with open(sh_filename, 'w') as sh_file:
        k = 0
        for i in range(c):  # 从0到c编号的csv
            for j in range(b):  # 每个编号的csv出现b次
                csv_file = f"{output_dir}/{str(i).zfill(2)}.csv"
                command = f"python3 {script_filename} {csv_file}"
                if (((k + 1) % a == 0)or((i+1)*(j+1)== b*c)):
                    sh_file.write(command + " \n")  # 每a行一次不加&
                else:
                    sh_file.write(command + " &\n")
                k += 1

if __name__ == '__main__':
    main()