# Generated by Django 4.0 on 2022-12-14 14:13

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0013_daqhat_channels'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='daqhat',
            name='boardNumber',
        ),
        migrations.RemoveField(
            model_name='daqhat',
            name='channels',
        ),
        migrations.AddField(
            model_name='daqhat',
            name='options',
            field=models.CharField(choices=[('DEFAULT', 'Default'), ('Continuous', 'Continuous')], default='', max_length=100),
        ),
        migrations.AlterField(
            model_name='daqhat',
            name='name',
            field=models.CharField(default='', max_length=50),
        ),
        migrations.AlterField(
            model_name='daqhat',
            name='scan_rate',
            field=models.FloatField(default=0),
        ),
    ]
