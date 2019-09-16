import csv, cv2, sys
sys.path.insert(0, './keras-retinanet-master')
import performDetection as PD

if __name__ == '__main__':
    out, csvOut = PD.retinanetDetection(r'C:\Users\Marc\Documents\egh455\Shark_whale - Evans 2016.11.18 F1 (2)_long.MP4')
    # out.release()
    # with open('results.csv', 'w', newline='') as csvFile:
    #     writer = csv.writer(csvFile)
    #     writer.writerows(csvOut)

    # csvFile.close()
    # cv2.destroyAllWindows()