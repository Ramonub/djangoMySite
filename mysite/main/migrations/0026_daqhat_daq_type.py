# Generated by Django 4.0 on 2022-12-19 13:03

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0025_analoginputrange_output_range'),
    ]

    operations = [
        migrations.AddField(
            model_name='daqhat',
            name='daq_type',
            field=models.IntegerField(choices=[('MCC_118', 'MCC_118'), ('MCC_128', 'MCC_128'), ('MCC_134', 'MCC_134'), ('MCC_152', 'MCC_152'), ('MCC_172', 'MCC_172')], default=0),
        ),
    ]
