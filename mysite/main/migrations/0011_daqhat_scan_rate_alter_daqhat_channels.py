# Generated by Django 4.0 on 2022-12-14 08:47

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0010_delete_person_daqhat_channels'),
    ]

    operations = [
        migrations.AddField(
            model_name='daqhat',
            name='scan_rate',
            field=models.IntegerField(null=True),
        ),
        migrations.AlterField(
            model_name='daqhat',
            name='channels',
            field=models.CharField(max_length=50, null=True),
        ),
    ]
