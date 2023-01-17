import xml.etree.cElementTree as ET


if __name__ == "__main__":

    my_sensor_list = [
                      {'name': 'OS_upper', 'unit': 'mu', 'chan': 0, 'type': 'capacitive', 'device_name': 'MadernRDMod3'},
                      {'name': 'OS_lower', 'unit': 'mu', 'chan': 1, 'type': 'capacitive', 'device_name': 'MadernRDMod3'},
                      {'name': 'DS_upper', 'unit': 'mu', 'chan': 2, 'type': 'capacitive', 'device_name': 'MadernRDMod3'},
                      {'name': 'DS_lower', 'unit': 'mu', 'chan': 3, 'type': 'capacitive', 'device_name': 'MadernRDMod3'},
                      {'name': 'OS_top', 'unit': 'mu', 'chan': 5, 'type': 'capacitive', 'device_name': 'MadernRDMod3'},
                      {'name': 'OS_mid', 'unit': 'mu', 'chan': 6, 'type': 'capacitive', 'device_name': 'MadernRDMod3'},
                      {'name': 'DS_top', 'unit': 'mu', 'chan': 7, 'type': 'capacitive', 'device_name': 'MadernRDMod3'},
                      {'name': 'DS_mid', 'unit': 'mu', 'chan': 8, 'type': 'capacitive', 'device_name': 'MadernRDMod3'},
                      {'name': 'OS_acc', 'unit': 'm/s^2', 'chan': 0, 'type': 'acceleration', 'device_name': 'MadernRDMod1'},
                      {'name': 'DS_acc', 'unit': 'm/s^2', 'chan': 1, 'type': 'acceleration', 'device_name': 'MadernRDMod1'},
                      {'name': 'Acc1_ax',  'unit': 'm/s^2', 'chan': 0, 'type': 'acceleration', 'device_name': 'MadernRDMod1'},
                      {'name': 'Acc2_rad', 'unit': 'm/s^2', 'chan': 1, 'type': 'acceleration', 'device_name': 'MadernRDMod1'},
                      ]

    # Root
    root = ET.Element('GabMeasurementSettings')

    # Sampling settings:
    daq_settings = ET.SubElement(root, 'daq_settings')

    chassis_name = ET.SubElement(daq_settings, 'chassis_name')
    chassis_name.text = 'MadernRD'

    chassis_name = ET.SubElement(daq_settings, 'task_name')
    chassis_name.text = 'Gap_task'

    sampling_rate = ET.SubElement(daq_settings, 'sampling_rate')
    sampling_rate.text = str(10240)

    buffer_size = ET.SubElement(daq_settings, 'buffer_size')
    buffer_size.text = str(200000)

    log_size = ET.SubElement(daq_settings, 'log_size')
    log_size.text = str(100000)


    ## --------- Sensor settings -----------------------
    sensor_settings = ET.SubElement(root, 'sensor_settings')

    for s in my_sensor_list:
        # Setting element:
        sensor_setting = ET.SubElement(sensor_settings, 'sensor')
        sensor_setting.set('name', s['name'])

        # Values:
        name = ET.SubElement(sensor_setting, 'name')
        name.text = s['name']

        module_name = ET.SubElement(sensor_setting, 'device_name')
        module_name.text = s['device_name']

        channel = ET.SubElement(sensor_setting, 'channel')
        channel.text = str(s['chan'])

        sensor_type = ET.SubElement(sensor_setting, 'type')
        sensor_type.text = s['type']

        unit = ET.SubElement(sensor_setting, 'unit')
        unit.text = s['unit']

    ## --------- Display Settings ---------------------
    display_settings = ET.SubElement(root, 'display_settings')


    tree = ET.ElementTree(root)
    fn = 'vibration_measurement_settings.xml'
    tree.write(fn)

    # Read:
    el = ET.parse(fn).getroot()
    for c in el.iter('sensor'):
        print('------{0}------'.format(c.attrib['name']))
