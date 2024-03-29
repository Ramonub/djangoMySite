# Generated by Django 4.0 on 2022-12-19 14:32

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0030_daqhat_input_range_daqhat_samples_per_channel_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='daqhat',
            name='daq_type',
            field=models.CharField(choices=[('MCC_118', 'MCC 118'), ('MCC_128', 'MCC 128'), ('MCC_134', 'MCC 134'), ('MCC_152', 'MCC 152'), ('MCC_172', 'MCC 172')], default='MCC 118', max_length=100),
        ),
        migrations.AlterField(
            model_name='daqhat',
            name='read_request_size',
            field=models.IntegerField(default=0),
        ),
        migrations.AlterField(
            model_name='daqhat',
            name='scan_rate',
            field=models.FloatField(default=0),
        ),
    ]
