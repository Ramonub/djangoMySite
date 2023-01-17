from Release import ftsensor
import time

class ftSensor():
    def __init__(self):
        self.sensor = ftsensor.ftSensorLib()
        res = -1
        fails = 0
        while res != 1:
            print('Attempting to configure...')
            res = self.sensor._configureStreamingBurst( 4, False, 0 )
            
            if res == 0:
                print('Succesfully configured')
                self.sensor._startStreamingThread()
                break
            else:
                fails +=1
                print(f'Failed {fails}')
                if fails > 10:
                    print('Failed configuring')
                    break
            time.sleep(0.1)

    def getNumberOfConnectedSensors(self):
        self.sensor._requestAllFTSensorsData()
        ftSensorsData = self.sensor._getFTSensorsData()
        # print(type(ftSensorsData.ftSensor.ChRaw.__long__()))
        # print(ftSensorsData.ftSensor.ChRaw)
        print(f'0 ', ftSensorsData.ftSensor.ft[0], 
                '\t1', ftSensorsData.ftSensor.ft[1], 
                '\t2', ftSensorsData.ftSensor.ft[2], 
                '\t3', ftSensorsData.ftSensor.ft[3],
                '\t4', ftSensorsData.ftSensor.ft[4], 
                '\t5', ftSensorsData.ftSensor.ft[5], flush=True)
        return self.sensor._getNumberOfConnectedSensors()

    def data(self):
        # while True:
        while True:
            self.sensor._requestAllFTSensorsData()
            ftSensorsData = self.sensor._getFTSensorsData()
            # print(type(ftSensorsData.ftSensor.ChRaw.__long__()))
            # print(ftSensorsData.ftSensor.ChRaw)
            print(f'0 ', ftSensorsData.ftSensor.ft[0], 
                  '\t1', ftSensorsData.ftSensor.ft[1], 
                  '\t2', ftSensorsData.ftSensor.ft[2], 
                  '\t3', ftSensorsData.ftSensor.ft[3],
                  '\t4', ftSensorsData.ftSensor.ft[4], 
                  '\t5', ftSensorsData.ftSensor.ft[5], flush=True)
            # for i, value in enumerate(ftSensorsData.ftSensor.ft):
            #     print(value)
            time.sleep(0.1)
