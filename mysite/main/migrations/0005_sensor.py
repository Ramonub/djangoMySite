# Generated by Django 4.1.2 on 2022-10-25 07:39

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0004_dreamreal_delete_person'),
    ]

    operations = [
        migrations.CreateModel(
            name='Sensor',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=50)),
                ('boardNumber', models.IntegerField()),
                ('channel', models.IntegerField()),
            ],
        ),
    ]